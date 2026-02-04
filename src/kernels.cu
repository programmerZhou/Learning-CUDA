#include <vector>
#include <cuda_fp16.h>
#include <limits>  // 用于获取不同类型的最小epsilon值

#include "../tester/utils.h"


template <typename T>
__global__ void reductionKernel(const T* d_input, size_t size, T* d_output) {
  // 共享内存用于存储各Warp的部分和
  __shared__ T s_warpSums[32];  // 32个Warp，每个Warp产生一个部分和
  
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  int warpId = threadIdx.x / 32;  // 当前线程所属的Warp ID (0-31)
  int laneId = threadIdx.x % 32;  // 当前线程在Warp内的ID (0-31)

  /* 第一级：每个Warp内部进行规约 */
  // 优化，使用规约加载模式
  T sum = T(0);
  int i = tid; 
  while(i < size) {
    sum += d_input[i];
    i += blockDim.x * gridDim.x;      // 网格级别的并行加载
  }
  // 完全展开 Warp Shuffle 操作
  sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
  sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
  sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
  sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
  sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);

  // 将每个Warp的部分和存入共享内存（由每个Warp的第一个线程完成）
  if (laneId == 0) {
      s_warpSums[warpId] = sum;
  }
  
  __syncthreads();  // 确保所有Warp部分和都已存入共享内存
  
  // 第二级：对Warp部分和进行规约
  sum = (warpId == 0 && laneId < 32) ? s_warpSums[laneId] : T(0);
  if (warpId == 0) {

    
      // 完全展开 Warp Shuffle 操作
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);
    
    // 将最终结果写入全局内存
    if (laneId == 0) {
        d_output[blockIdx.x] = sum;
    }
  }
}

/**
 * @brief Computes the trace of a matrix.
 *
 * The trace of a matrix is defined as the sum of its diagonal elements.
 * This function expects a flattened row-major matrix stored in a
 * std::vector. If the matrix is not square, the trace will sum up
 * elements along the main diagonal up to the smaller of rows or cols.
 *
 * @tparam T The numeric type of matrix elements (e.g., float, int).
 * @param h_input A flattened matrix of size rows * cols.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return The trace (sum of diagonal values) of the matrix.
 */

#define TRACE_BLOCK_SIZE 256 

#define CHECK_CUDA(call)                                                   \
  {                                                                      \
    cudaError_t err = call;                                              \
    if (err != cudaSuccess) {                                            \
      fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n",      \
              __FILE__, __LINE__, cudaGetErrorString(err));              \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  }

template <typename T>
T trace_cpu(const std::vector<T>& h_input, size_t rows, size_t cols) {  // 添加cols参数
  size_t min_dim = (rows < cols) ? rows : cols;
  T trace_sum = 0;
  for (size_t i = 0; i < min_dim; ++i) {
    trace_sum += h_input[i * cols + i];  // 使用cols计算索引
  }
  return trace_sum;
}

/* 
 * 将对角线上的数据拷贝到新的区域的实现版本
 */
template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
  
  size_t min_dim = (rows < cols) ? rows : cols;
  
  /*对于小矩形，直接使用CPU版本更高效*/
  if (min_dim <= 512) {
    return trace_cpu(h_input, rows, cols);
  }

  // allocate device memory
  T * d_input;

  int bytes = min_dim * sizeof(T);
  CHECK_CUDA(cudaMalloc(&d_input, bytes));

  std::vector<T> h_diagonal(min_dim);
  for (size_t i = 0; i < min_dim; ++i) {  
    h_diagonal[i] = h_input[i * cols + i];
  }

  CHECK_CUDA(cudaMemcpy(d_input, h_diagonal.data(), bytes, cudaMemcpyHostToDevice));

  // Convert computation trace into the sum of computation arrays.
  // Use reduction method for computation.
  dim3 grid ((min_dim / 2 + TRACE_BLOCK_SIZE - 1) / TRACE_BLOCK_SIZE);
  dim3 block (TRACE_BLOCK_SIZE);
  T * d_output; 
  
  size_t output_bytes = grid.x * sizeof(T); 
  CHECK_CUDA(cudaMalloc(&d_output, output_bytes));
  CHECK_CUDA(cudaMemset(d_output, 0, output_bytes)); 
  
  reductionKernel<<<grid, block>>>(d_input, min_dim, d_output);

  // copy result back to host
  std::vector<T> h_output(grid.x);
  CHECK_CUDA(cudaMemcpy(h_output.data(), d_output, output_bytes, cudaMemcpyDeviceToHost)); 
  T trace_sum = 0;
  for (size_t i = 0; i < grid.x; ++i) {
    trace_sum += h_output[i];
  }

  // free device memory
  CHECK_CUDA(cudaFree(d_input));
  CHECK_CUDA(cudaFree(d_output));
  return trace_sum;
}


// 定义块大小参数
#define QUERY_BLOCK_SIZE 128  // Query 块大小
#define KEY_BLOCK_SIZE 64     // Key/Value 块大小

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 * 
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length  
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention headg
 * @param[in] is_causal Whether to apply causal masking
 */
template <typename T>
void selfAttentionCPU(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) { 
  
  // 计算输出张量大小
  const size_t output_size = static_cast<size_t>(batch_size * target_seq_len * query_heads * head_dim);
  
  // 检查输入有效性
  if (h_q.size() != output_size ||
      h_k.size() != static_cast<size_t>(batch_size * src_seq_len * kv_heads * head_dim) ||
      h_v.size() != static_cast<size_t>(batch_size * src_seq_len * kv_heads * head_dim)) {
      throw std::invalid_argument("Input tensor sizes do not match dimensions");
  }
  
  // 常量
  const float scale = 1.0 / sqrt(head_dim);  // 缩放因子
  // printf("Scale factor: %f\n", scale);
  const float epsilon = 1e-12f; 
  
  // GQA分组大小
  int group_size = query_heads / kv_heads;
  if (query_heads % kv_heads != 0) {
    throw std::invalid_argument("query_heads must be divisible by kv_heads for grouped query attention");
  }
  
  // 索引函数
  auto idx_q = [&](int b, int t, int h, int d) {
    return ((b * target_seq_len + t) * query_heads + h) * head_dim + d;
  };
  
  auto idx_kv = [&](int b, int s, int h, int d) {
    return ((b * src_seq_len + s) * kv_heads + h) * head_dim + d;
  };
  
  auto idx_o = [&](int b, int t, int h, int d) {
    return ((b * target_seq_len + t) * query_heads + h) * head_dim + d;
  };
  
  // 简单的四重循环实现
  for (int b = 0; b < batch_size; ++b) {  // 批次循环
    for (int t = 0; t < target_seq_len; ++t) {  // 目标序列循环
      for (int qh = 0; qh < query_heads; ++qh) {  // Query头循环
        // GQA映射：确定对应的KV头
        int kvh = qh / group_size;
        
        // 初始化输出向量
        std::vector<T> output(head_dim, static_cast<T>(0));
        
        // 计算注意力权重 - 所有中间计算都使用float类型
        std::vector<float> attention_scores(src_seq_len, 0.0f);
        float max_attn_float = -1e10f;
        
        // 第一步：计算所有注意力分数
        for (int s = 0; s < src_seq_len; ++s) {  // 源序列循环
          // 因果掩码检查：如果是因果注意力且目标位置在源位置之前
          if (is_causal && t < s) {
            attention_scores[s] = -1e10f;  // 使用float的负无穷
            continue;
          }
            
          // 计算点积：Qi · Kj - 使用double提高精度
          float dot_product = 0.0;
          for (int d = 0; d < head_dim; ++d) {
            float q_val = h_q[idx_q(b, t, qh, d)];
            float k_val = h_k[idx_kv(b, s, kvh, d)];
            dot_product += q_val * k_val;
          }
          
          // 应用缩放因子
          float score = dot_product * scale;

          attention_scores[s] = score;  // 转换为float存储
          
          // 跟踪最大值用于数值稳定性 - 使用float
          if (score > max_attn_float) {
            max_attn_float = score;
          }
        }
      
        // 第二步：计算softmax
        float sum_exp = 0.0;
      
        std::vector<float> attention_weights(src_seq_len);
        for (int s = 0; s < src_seq_len; ++s) {
          // 计算exp(score - max_attn)以提高数值稳定性
          float exp_val = exp(attention_scores[s] - max_attn_float);
          attention_weights[s] = exp_val;
          sum_exp += exp_val;
        }
          
        for (int s = 0; s < src_seq_len; ++s) {
          attention_weights[s] = (attention_weights[s]) / (sum_exp + epsilon);
        }
          
        for (int s = 0; s < src_seq_len; ++s) {
          float weight = attention_weights[s];
          
          // 将注意力权重与值向量相乘并累加
          for (int d = 0; d < head_dim; ++d) {
            T v_val = h_v[idx_kv(b, s, kvh, d)];
            output[d] += T(weight * float(v_val));
          }
        }
        
        // 第四步：写入输出
        for (int d = 0; d < head_dim; ++d) {
          h_o[idx_o(b, t, qh, d)] = output[d];
        }
      }
    }
  }
}

/**
 * @brief CUDA内核：实现注意力机制计算
 * 
 * @tparam T Data type (float) for input/output tensors
 * @param d_q Query tensor on device
 * @param d_k Key tensor on device
 * @param d_v Value tensor on device
 * @param d_o Output tensor on device
 * @param batch_size Batch dimension size
 * @param target_seq_len Target sequence length
 * @param src_seq_len Source sequence length
 * @param query_heads Number of query attention heads
 * @param kv_heads Number of key/value heads
 * @param head_dim Dimension size of each attention head
 * @param is_causal Whether to apply causal masking
 */
template <typename T>
__global__ void selfAttentionKernel(const T* __restrict__ d_q, 
                                   const T* __restrict__ d_k, 
                                   const T* __restrict__ d_v, 
                                   T* __restrict__ d_o, 
                                   int batch_size, 
                                   int target_seq_len, 
                                   int src_seq_len, 
                                   int query_heads, 
                                   int kv_heads, 
                                   int head_dim, 
                                   bool is_causal) {
    // 每个线程块处理一个 (batch, query_seq, query_head) 组合
    // 线程块索引
    int batch_idx = blockIdx.z;
    int query_head_idx = blockIdx.y;
    int query_seq_idx = blockIdx.x;
    
    // 检查线程块是否在有效范围内
    if (batch_idx >= batch_size || query_head_idx >= query_heads || 
        query_seq_idx >= target_seq_len) {
        return;
    }
    
    // GQA映射：确定对应的KV头
    int group_size = query_heads / kv_heads;
    int kv_head_idx = query_head_idx / group_size;
    
    // 计算索引函数
    auto idx_q = [&](int b, int t, int h, int d) {
        return ((b * target_seq_len + t) * query_heads + h) * head_dim + d;
    };
    
    auto idx_kv = [&](int b, int s, int h, int d) {
        return ((b * src_seq_len + s) * kv_heads + h) * head_dim + d;
    };
    
    auto idx_o = [&](int b, int t, int h, int d) {
        return ((b * target_seq_len + t) * query_heads + h) * head_dim + d;
    };
    
    // 共享内存：存储当前query向量
    __shared__ T s_q[128];  // head_dim最大支持128
    
    // 所有线程加载query向量到共享内存
    if (threadIdx.x < head_dim) {
        s_q[threadIdx.x] = d_q[idx_q(batch_idx, query_seq_idx, query_head_idx, threadIdx.x)];
    }
    __syncthreads();
    
    // 计算缩放因子
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    const float epsilon = 1e-12f;
    
    // 第一步：计算所有注意力分数的最大值（用于数值稳定性）
    float max_attn = -1e10f;
    for (int src_seq_idx = 0; src_seq_idx < src_seq_len; ++src_seq_idx) {
        // 因果掩码检查
        if (is_causal && query_seq_idx < src_seq_idx) {
            continue;
        }
        
        // 计算点积：Qi · Kj
        float dot_product = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            T q_val = s_q[d];  // 从共享内存读取，减少全局访问
            T k_val = d_k[idx_kv(batch_idx, src_seq_idx, kv_head_idx, d)];
            dot_product += static_cast<float>(q_val) * static_cast<float>(k_val);
        }
        
        // 应用缩放因子
        float score = dot_product * scale;
        
        // 更新最大值用于数值稳定性
        if (score > max_attn) {
            max_attn = score;
        }
    }
    
    // 第二步：计算所有注意力分数的指数和
    float sum_exp = 0.0f;
    for (int src_seq_idx = 0; src_seq_idx < src_seq_len; ++src_seq_idx) {
        // 因果掩码检查
        if (is_causal && query_seq_idx < src_seq_idx) {
            continue;
        }
        
        // 重新计算点积和缩放
        float dot_product = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            T q_val = s_q[d];
            T k_val = d_k[idx_kv(batch_idx, src_seq_idx, kv_head_idx, d)];
            dot_product += static_cast<float>(q_val) * static_cast<float>(k_val);
        }
        float score = dot_product * scale;
        
        // 计算指数并累加
        sum_exp += expf(score - max_attn);
    }
    
    // 归一化因子
    float inv_sum_exp = 1.0f / (sum_exp + epsilon);
    
    // 第三步：计算输出
    // 每个线程负责一个维度
    if (threadIdx.x < head_dim) {
        // 使用float进行累加，提高精度（特别是对于half类型）
        float output_val_float = 0.0f;
        for (int src_seq_idx = 0; src_seq_idx < src_seq_len; ++src_seq_idx) {
            // 因果掩码检查
            if (is_causal && query_seq_idx < src_seq_idx) {
                continue;
            }
            
            // 重新计算点积、缩放和softmax
            float dot_product = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                T q_val = s_q[d];
                T k_val = d_k[idx_kv(batch_idx, src_seq_idx, kv_head_idx, d)];
                dot_product += static_cast<float>(q_val) * static_cast<float>(k_val);
            }
            float score = dot_product * scale;
            float softmax_attn = expf(score - max_attn) * inv_sum_exp;
            
            // 加载值并累加（使用float提高精度）
            T v_val = d_v[idx_kv(batch_idx, src_seq_idx, kv_head_idx, threadIdx.x)];
            output_val_float += softmax_attn * static_cast<float>(v_val);
        }
        
        // 写入输出，转换回目标类型
        d_o[idx_o(batch_idx, query_seq_idx, query_head_idx, threadIdx.x)] = static_cast<T>(output_val_float);
    }
}

/**
 * @brief Computes attention for given query, key, and value tensors on GPU.
 * 
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length  
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention headg
 * @param[in] is_causal Whether to apply causal masking
 */
template <typename T>
void selfAttentionGPU(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {

    // 计算张量大小
    const size_t q_size = batch_size * target_seq_len * query_heads * head_dim;
    const size_t kv_size = batch_size * src_seq_len * kv_heads * head_dim;
    const size_t o_size = q_size;
    
    // 分配设备内存
    T* d_q, *d_k, *d_v, *d_o;
    CHECK_CUDA(cudaMalloc(&d_q, q_size * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_k, kv_size * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_v, kv_size * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_o, o_size * sizeof(T)));
    
    // 复制数据到设备
    CHECK_CUDA(cudaMemcpy(d_q, h_q.data(), q_size * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_k, h_k.data(), kv_size * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_v, h_v.data(), kv_size * sizeof(T), cudaMemcpyHostToDevice));
    
    // 配置线程块和网格
    dim3 block(head_dim, 1, 1);  // 每个线程处理一个维度
    dim3 grid(target_seq_len, query_heads, batch_size);  // 每个查询序列和头一个块
    
    // 启动内核
    selfAttentionKernel<<<grid, block>>>(d_q, d_k, d_v, d_o, 
                                       batch_size, target_seq_len, src_seq_len, 
                                       query_heads, kv_heads, head_dim, is_causal);
    
    // 检查内核启动错误
    CHECK_CUDA(cudaGetLastError());
    
    // 同步设备
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // 复制结果回主机
    h_o.resize(o_size);
    CHECK_CUDA(cudaMemcpy(h_o.data(), d_o, o_size * sizeof(T), cudaMemcpyDeviceToHost));
    
    // 释放设备内存
    CHECK_CUDA(cudaFree(d_q));
    CHECK_CUDA(cudaFree(d_k));
    CHECK_CUDA(cudaFree(d_v));
    CHECK_CUDA(cudaFree(d_o));
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 * 
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length  
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention headg
 * @param[in] is_causal Whether to apply causal masking
 */
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {
  // 调用CPU实现
  selfAttentionGPU(h_q, h_k, h_v, h_o,
                  batch_size, target_seq_len, src_seq_len,
                  query_heads, kv_heads, head_dim, is_causal);
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int trace<int>(const std::vector<int>&, size_t, size_t);
template float trace<float>(const std::vector<float>&, size_t, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);