#include <vector>
#include <cuda_fp16.h>

#include "../tester/utils.h"


template <typename T>
__global__ void reductionKernel(const T* d_input, size_t size, T* d_output) {
    // 共享内存用于存储各Warp的部分和
    __shared__ T s_warpSums[32];  // 32个Warp，每个Warp产生一个部分和
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
  
    int warpId = threadIdx.x / 32;  // 当前线程所属的Warp ID (0-31)
    int laneId = threadIdx.x % 32;  // 当前线程在Warp内的ID (0-31)

    // 第一级：Warp内部规约 - 使用Shuffle指令
    T sum = T(0);
    if (tid < size) {
        sum = d_input[tid];
        
       
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
    // 只使用第一个Warp（32个线程）来处理32个Warp部分和
    if (warpId == 0 && laneId < 32) {
        if (laneId < 32) {
            sum = s_warpSums[laneId];
        } else {
            sum = T(0);
        }
        
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

template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
  
  size_t min_dim = (rows < cols) ? rows : cols;
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
  dim3 grid ((min_dim + TRACE_BLOCK_SIZE - 1) / TRACE_BLOCK_SIZE);
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

// // CUDA 核函数：FlashAttention 实现
// template <typename T>
// __global__ void flashAttentionKernel(
//     const T* __restrict__ d_q,  // [batch_size, target_seq_len, query_heads, head_dim]
//     const T* __restrict__ d_k,  // [batch_size, src_seq_len, kv_heads, head_dim]
//     const T* __restrict__ d_v,  // [batch_size, src_seq_len, kv_heads, head_dim]
//     T* __restrict__ d_o,        // [batch_size, target_seq_len, query_heads, head_dim]
//     int batch_size,
//     int target_seq_len,
//     int src_seq_len,
//     int query_heads,
//     int kv_heads,
//     int head_dim,
//     bool is_causal,
//     int group_size) {  // GQA 分组大小 (query_heads / kv_heads)

//     // 共享内存：存储 Query、Key、Value 块
//     __shared__ T s_q[QUERY_BLOCK_SIZE][128];   // head_dim 最大支持 128
//     __shared__ T s_k[KEY_BLOCK_SIZE][128];
//     __shared__ T s_v[KEY_BLOCK_SIZE][128];

//     // 线程索引
//     int dim = threadIdx.x;                     // head_dim 维度
//     int query_idx_in_block = threadIdx.y;      // Query 块内序列索引
//     int batch_head_idx = blockIdx.z;           // batch 和 query_head 组合索引
//     int query_block_idx = blockIdx.y;          // Query 块索引

//     // 计算实际的 batch 和 query_head 索引
//     int batch_idx = batch_head_idx / query_heads;
//     int query_head_idx = batch_head_idx % query_heads;

//     // 计算对应的 KV head 索引 (处理 GQA)
//     int kv_head_idx = query_head_idx / group_size;

//     // 计算当前 Query 块的边界
//     int query_start = query_block_idx * QUERY_BLOCK_SIZE;
//     int query_end = min(query_start + QUERY_BLOCK_SIZE, target_seq_len);
//     int query_len_in_block = query_end - query_start;

//     // 检查当前线程是否在有效范围内
//     if (query_idx_in_block >= query_len_in_block || dim >= head_dim) {
//         return;
//     }

//     // 当前处理的 Query 序列索引
//     int query_seq_idx = query_start + query_idx_in_block;

//     // 加载 Query 到共享内存
//     size_t q_idx = ((batch_idx * target_seq_len + query_seq_idx) * query_heads + query_head_idx) * head_dim + dim;
//     s_q[query_idx_in_block][dim] = d_q[q_idx];
//     __syncthreads();

//     // 初始化输出累加器和 softmax 中间变量
//     T o_accum = 0.0f;
//     T max_attn = -1e10f;
//     T sum_exp_attn = 0.0f;

//     // 遍历所有 Key 块
//     for (int key_block_idx = 0; key_block_idx < ((src_seq_len + KEY_BLOCK_SIZE - 1) / KEY_BLOCK_SIZE); key_block_idx++) {
//         // 计算当前 Key 块的边界
//         int key_start = key_block_idx * KEY_BLOCK_SIZE;
//         int key_end = min(key_start + KEY_BLOCK_SIZE, src_seq_len);
//         int key_len_in_block = key_end - key_start;

//         // 加载 Key 和 Value 到共享内存
//         int key_idx_in_block = threadIdx.y;
//         if (key_idx_in_block < key_len_in_block && dim < head_dim) {
//             int key_seq_idx = key_start + key_idx_in_block;
//             size_t k_idx = ((batch_idx * src_seq_len + key_seq_idx) * kv_heads + kv_head_idx) * head_dim + dim;
//             size_t v_idx = ((batch_idx * src_seq_len + key_seq_idx) * kv_heads + kv_head_idx) * head_dim + dim;
//             s_k[key_idx_in_block][dim] = d_k[k_idx];
//             s_v[key_idx_in_block][dim] = d_v[v_idx];
//         }
//         __syncthreads();

//         // 计算当前 Query 块和 Key 块的注意力
//         for (int key_idx_in_block = 0; key_idx_in_block < key_len_in_block; key_idx_in_block++) {
//             int key_seq_idx = key_start + key_idx_in_block;

//             // 因果掩码检查
//             if (is_causal && query_seq_idx < key_seq_idx) {
//                 continue;
//             }

//             // 计算 Q*K^T (点积)
//             T attn_val = 0.0f;
//             for (int d = 0; d < head_dim; d++) {
//                 attn_val += s_q[query_idx_in_block][d] * s_k[key_idx_in_block][d];
//             }

//             // 缩放 (1/sqrt(head_dim))
//             attn_val /= sqrtf(static_cast<float>(head_dim));

//             // 更新 softmax 中间变量
//             if (attn_val > max_attn) {
//                 sum_exp_attn = expf(attn_val - max_attn) * sum_exp_attn;
//                 max_attn = attn_val;
//             } else {
//                 sum_exp_attn += expf(attn_val - max_attn);
//             }
//         }
//         __syncthreads();
//     }

//     // 重新遍历所有 Key 块，计算最终输出
//     for (int key_block_idx = 0; key_block_idx < ((src_seq_len + KEY_BLOCK_SIZE - 1) / KEY_BLOCK_SIZE); key_block_idx++) {
//         // 计算当前 Key 块的边界
//         int key_start = key_block_idx * KEY_BLOCK_SIZE;
//         int key_end = min(key_start + KEY_BLOCK_SIZE, src_seq_len);
//         int key_len_in_block = key_end - key_start;

//         // 加载 Key 和 Value 到共享内存
//         int key_idx_in_block = threadIdx.y;
//         if (key_idx_in_block < key_len_in_block && dim < head_dim) {
//             int key_seq_idx = key_start + key_idx_in_block;
//             size_t k_idx = ((batch_idx * src_seq_len + key_seq_idx) * kv_heads + kv_head_idx) * head_dim + dim;
//             size_t v_idx = ((batch_idx * src_seq_len + key_seq_idx) * kv_heads + kv_head_idx) * head_dim + dim;
//             s_k[key_idx_in_block][dim] = d_k[k_idx];
//             s_v[key_idx_in_block][dim] = d_v[v_idx];
//         }
//         __syncthreads();

//         // 计算注意力权重和 Value 的乘积
//         for (int key_idx_in_block = 0; key_idx_in_block < key_len_in_block; key_idx_in_block++) {
//             int key_seq_idx = key_start + key_idx_in_block;

//             // 因果掩码检查
//             if (is_causal && query_seq_idx < key_seq_idx) {
//                 continue;
//             }

//             // 计算 Q*K^T
//             T attn_val = 0.0f;
//             for (int d = 0; d < head_dim; d++) {
//                 attn_val += s_q[query_idx_in_block][d] * s_k[key_idx_in_block][d];
//             }

//             // 缩放和 softmax
//             attn_val /= sqrtf(static_cast<float>(head_dim));
//             T softmax_attn = expf(attn_val - max_attn) / sum_exp_attn;

//             // 累加 Value 贡献
//             o_accum += softmax_attn * s_v[key_idx_in_block][dim];
//         }
//         __syncthreads();
//     }

//     // 将结果写回全局内存
//     size_t o_idx = ((batch_idx * target_seq_len + query_seq_idx) * query_heads + query_head_idx) * head_dim + dim;
//     d_o[o_idx] = o_accum;
// }

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
  /* 
    实现FlashAttention算法，支持分组查询注意力（GQA）。
    输入：
      h_q: 查询张量，形状为 [batch_size, target_seq_len, query_heads, head_dim]
      h_k: 键张量，形状为 [batch_size, src_seq_len, kv_heads, head_dim]
      h_v: 值张量，形状为 [batch_size, src_seq_len, kv_heads, head_dim]
    输出：
      h_o: 输出注意力张量，形状为 [batch_size, target_seq_len, query_heads, head_dim]
    参数：
      batch_size: 批次大小
      target_seq_len: 目标序列长度
      src_seq_len: 源序列长度
      query_heads: 查询头数
      kv_heads: 键/值头数
      head_dim: 每个注意力头的维度大小
      is_causal: 是否应用因果掩码

    总特征维度 = 头数 × 头维度
    即：C = H × d
    其中：
    C：总特征维度（channel dimension）
    H：注意力头数（number of heads）  
    d：每个头的维度（head dimension，通常记为d_k, d_v）

    分组查询注意力（GQA）
    配置: query_heads = H, kv_heads = G (1 < G < H)
    特点: 查询头分成G组，每组共享K、V投影
    内存: 适中 (H+2G个投影矩阵)
    质量: 接近MHA
  */

  /*打印出参数的形状*/
  printf("batch_size: %d, target_seq_len: %d, src_seq_len: %d, query_heads: %d, kv_heads: %d, head_dim: %d, is_causal: %d\n",
          batch_size, target_seq_len, src_seq_len, query_heads, kv_heads, head_dim, is_causal);
  // 检查输入有效性
  if (h_q.size() != static_cast<size_t>(batch_size * target_seq_len * query_heads * head_dim) ||
      h_k.size() != static_cast<size_t>(batch_size * src_seq_len * kv_heads * head_dim) ||
      h_v.size() != static_cast<size_t>(batch_size * src_seq_len * kv_heads * head_dim)) {
      throw std::invalid_argument("Input tensor sizes do not match dimensions");
  }
  // 常量
  const T scale = static_cast<T>(1.0 / sqrt(static_cast<double>(head_dim)));
  const T neg_inf = static_cast<T>(-1e10);  // 用于掩码的负无穷
    
  // GQA分组大小
  int group_size = query_heads / kv_heads;
  if (query_heads % kv_heads != 0) {
    throw std::invalid_argument("query_heads must be divisible by kv_heads for grouped query attention");
  }
    
    // 索引函数 - NHWC布局
    auto idx_q = [&](int b, int t, int h, int d) {
        return ((b * target_seq_len + t) * query_heads + h) * head_dim + d;
    };
    
    auto idx_kv = [&](int b, int s, int h, int d) {
        return ((b * src_seq_len + s) * kv_heads + h) * head_dim + d;
    };
    
    auto idx_o = [&](int b, int t, int h, int d) {
        return ((b * target_seq_len + t) * query_heads + h) * head_dim + d;
    };
    
    // 设置分块大小（根据head_dim调整，通常让分块适合缓存）
    const int Br = 64;  // Q的分块大小（行方向）
    const int Bc = 128; // K/V的分块大小（列方向）
    
    // 主循环
    for (int b = 0; b < batch_size; ++b) {
        for (int qh = 0; qh < query_heads; ++qh) {
            int kvh = qh / group_size;  // GQA（）映射 分组查询注意力
            
            // 为每个query头分配输出统计量
            std::vector<T> Oi(head_dim, static_cast<T>(0));  // 输出累加器
            T mi = static_cast<T>(-1e10);  // 最大值
            T li = static_cast<T>(0);      // 指数和
            
            // 分块处理K/V
            for (int block_c = 0; block_c < src_seq_len; block_c += Bc) {
                int block_c_end = fminf(block_c + Bc, src_seq_len);
                
                // 1. 加载当前KV块
                std::vector<T> Kj((block_c_end - block_c) * head_dim);
                std::vector<T> Vj((block_c_end - block_c) * head_dim);

                for (int s = block_c; s < block_c_end; ++s) {
                    int offset = (s - block_c) * head_dim;
                    for (int d = 0; d < head_dim; ++d) {
                        Kj[offset + d] = h_k[idx_kv(b, s, kvh, d)];  // ✅ 正确索引
                        Vj[offset + d] = h_v[idx_kv(b, s, kvh, d)];
                    }
                }
                
                // 2. 分块处理Q（行方向）
                for (int block_r = 0; block_r < target_seq_len; block_r += Br) {
                    int block_r_end = fminf(block_r + Br, target_seq_len);
                    
                    for (int t = block_r; t < block_r_end; ++t) {
                        // 加载当前Q向量
                        std::vector<T> Qi(head_dim);
                        for (int d = 0; d < head_dim; ++d) {
                            Qi[d] = h_q[idx_q(b, t, qh, d)];
                        }
                        
                        // 为当前query初始化局部统计
                        T m_ij = static_cast<T>(-1e10);
                        std::vector<T> Pij(block_c_end - block_c, static_cast<T>(0));
                        std::vector<T> Oij(head_dim, static_cast<T>(0));
                        
                        // 3. 计算Sij = Qi * Kj^T
                        for (int sj = block_c; sj < block_c_end; ++sj) {
                            int j = sj - block_c;
                            
                            // 检查因果掩码
                            if (is_causal && t < sj) {
                                Pij[j] = neg_inf;
                                continue;
                            }
                            
                            // 计算点积
                            T dot = static_cast<T>(0);
                            for (int d = 0; d < head_dim; ++d) {
                                dot = static_cast<T>(static_cast<float>(dot) + static_cast<float>(Qi[d]) * static_cast<float>(Kj[j * head_dim + d]));
                            }
                            dot = static_cast<T>(static_cast<float>(dot) * static_cast<float>(scale));
                            
                            Pij[j] = dot;
                            m_ij = static_cast<T>(fmaxf(static_cast<float>(m_ij), static_cast<float>(dot)));
                        }
                        
                        // 4. 计算Pij = exp(Sij - m_ij)
                        T lij = static_cast<T>(0);
                        for (int j = 0; j < block_c_end - block_c; ++j) {
                            if (static_cast<float>(Pij[j]) == static_cast<float>(neg_inf)) {
                                Pij[j] = static_cast<T>(0);
                            } else {
                                Pij[j] = static_cast<T>(exp(static_cast<float>(Pij[j]) - static_cast<float>(m_ij)));
                                lij = static_cast<T>(static_cast<float>(lij) + static_cast<float>(Pij[j]));
                            }
                        }
                        
                        // 5. 在线softmax重归一化
                        T mi_new = static_cast<T>(fmaxf(static_cast<float>(mi), static_cast<float>(m_ij)));
                        T scale_o = static_cast<T>(exp(static_cast<float>(mi) - static_cast<float>(mi_new)));
                        T scale_ij = static_cast<T>(exp(static_cast<float>(m_ij) - static_cast<float>(mi_new)));
                        
                        // 更新输出累加器
                        for (int d = 0; d < head_dim; ++d) {
                            Oi[d] = static_cast<T>(static_cast<float>(Oi[d]) * static_cast<float>(scale_o));
                        }
                        
                        // 添加当前块的贡献
                        for (int j = 0; j < block_c_end - block_c; ++j) {
                            if (static_cast<float>(Pij[j]) > static_cast<float>(0)) {
                                for (int d = 0; d < head_dim; ++d) {
                                    Oi[d] = static_cast<T>(static_cast<float>(Oi[d]) + static_cast<float>(Pij[j]) * static_cast<float>(Vj[j * head_dim + d]));
                                }
                            }
                        }
                        
                        // 更新统计量
                        li = static_cast<T>(static_cast<float>(li) * static_cast<float>(scale_o) + static_cast<float>(lij) * static_cast<float>(scale_ij));
                        mi = mi_new;
                        
                        // 如果这是最后一个K/V块，写入最终输出
                        if (block_c + Bc >= src_seq_len) {
                            for (int d = 0; d < head_dim; ++d) {
                                h_o[idx_o(b, t, qh, d)] = static_cast<T>(static_cast<float>(Oi[d]) / static_cast<float>(li));
                            }
                            // 重置统计量用于下一个query
                            if (t == block_r_end - 1 && block_r + Br >= target_seq_len) {
                                std::fill(Oi.begin(), Oi.end(), static_cast<T>(0));
                                mi = static_cast<T>(-1e10);
                                li = static_cast<T>(0);
                            }
                        }
                    }
                }
            }
        }
    }

}

// template <typename T>
// void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
//                     const std::vector<T>& h_v, std::vector<T>& h_o,
//                     int batch_size, int target_seq_len, int src_seq_len, 
//                     int query_heads, int kv_heads, int head_dim, bool is_causal) { 
    
//     // 参数检查和初始化...
    
//     const T scale = static_cast<T>(1.0 / sqrt(static_cast<double>(head_dim)));
//     const T neg_inf = static_cast<T>(-1e10);
//     int group_size = query_heads / kv_heads;
    
//     // 索引函数
//     auto idx_q = [&](int b, int t, int h, int d) {
//         return ((b * target_seq_len + t) * query_heads + h) * head_dim + d;
//     };
    
//     auto idx_kv = [&](int b, int s, int h, int d) {
//         return ((b * src_seq_len + s) * kv_heads + h) * head_dim + d;
//     };
    
//     auto idx_o = [&](int b, int t, int h, int d) {
//         return ((b * target_seq_len + t) * query_heads + h) * head_dim + d;
//     };
    
//     const int Br = 64;  // Q分块
//     const int Bc = 128; // KV分块
    
//     // 主循环
//     for (int b = 0; b < batch_size; ++b) {
//         for (int qh = 0; qh < query_heads; ++qh) {
//             int kvh = qh / group_size;  // ✅ GQA映射：多个qh映射到同一个kvh
            
//             // 为当前查询头分配统计量（每个target位置应该独立！）
//             // ❌ 当前代码问题：Oi, mi, li应该在每个t位置独立
//             // ✅ 应该为每个t位置维护独立的统计量
            
//             // 存储每个目标位置的统计量
//             std::vector<std::vector<T>> Oi_list(target_seq_len, std::vector<T>(head_dim, 0));
//             std::vector<T> mi_list(target_seq_len, static_cast<T>(-1e10));
//             std::vector<T> li_list(target_seq_len, static_cast<T>(0));
            
//             // 分块处理K/V
//             for (int block_c = 0; block_c < src_seq_len; block_c += Bc) {
//                 int block_c_end = std::min(block_c + Bc, src_seq_len);
//                 int block_c_size = block_c_end - block_c;
                
//                 // ✅ 修正：正确加载K和V块
//                 std::vector<T> Kj(block_c_size * head_dim);
//                 std::vector<T> Vj(block_c_size * head_dim);
                
//                 for (int s = block_c; s < block_c_end; ++s) {
//                     int offset = (s - block_c) * head_dim;
//                     for (int d = 0; d < head_dim; ++d) {
//                         Kj[offset + d] = h_k[idx_kv(b, s, kvh, d)];
//                         Vj[offset + d] = h_v[idx_kv(b, s, kvh, d)];
//                     }
//                 }
                
//                 // 处理Q分块
//                 for (int block_r = 0; block_r < target_seq_len; block_r += Br) {
//                     int block_r_end = std::min(block_r + Br, target_seq_len);
                    
//                     for (int t = block_r; t < block_r_end; ++t) {
//                         // 加载当前Q向量
//                         std::vector<T> Qi(head_dim);
//                         for (int d = 0; d < head_dim; ++d) {
//                             Qi[d] = h_q[idx_q(b, t, qh, d)];
//                         }
                        
//                         // 获取当前t位置的统计量
//                         std::vector<T>& Oi = Oi_list[t];
//                         T& mi = mi_list[t];
//                         T& li = li_list[t];
                        
//                         // 计算当前块的点积
//                         std::vector<T> Sij(block_c_size);
//                         T mij = static_cast<T>(-1e10);
                        
//                         for (int j = 0; j < block_c_size; ++j) {
//                             int sj = block_c + j;
                            
//                             // 因果掩码
//                             if (is_causal && t < sj) {
//                                 Sij[j] = neg_inf;
//                                 continue;
//                             }
                            
//                             // 计算点积
//                             T dot = static_cast<T>(0);
//                             const T* Kj_ptr = &Kj[j * head_dim];
//                             for (int d = 0; d < head_dim; ++d) {
//                                 dot += Qi[d] * Kj_ptr[d];
//                             }
//                             Sij[j] = dot * scale;
//                             mij = std::max(mij, Sij[j]);
//                         }
                        
//                         // 计算当前块的指数和
//                         T lij = static_cast<T>(0);
//                         std::vector<T> Pij(block_c_size, static_cast<T>(0));
                        
//                         for (int j = 0; j < block_c_size; ++j) {
//                             if (Sij[j] > neg_inf) {
//                                 Pij[j] = std::exp(Sij[j] - mij);
//                                 lij += Pij[j];
//                             }
//                         }
                        
//                         // 在线softmax更新
//                         T mi_new = std::max(mi, mij);
//                         T scale_o = (mi > -1e9f) ? std::exp(mi - mi_new) : static_cast<T>(0);
//                         T scale_ij = std::exp(mij - mi_new);
                        
//                         // 更新输出
//                         for (int d = 0; d < head_dim; ++d) {
//                             Oi[d] = Oi[d] * scale_o;
//                         }
                        
//                         // 添加当前块的贡献
//                         for (int j = 0; j < block_c_size; ++j) {
//                             if (Pij[j] > static_cast<T>(0)) {
//                                 const T* Vj_ptr = &Vj[j * head_dim];
//                                 T pij_scaled = Pij[j] * scale_ij;
//                                 for (int d = 0; d < head_dim; ++d) {
//                                     Oi[d] += pij_scaled * Vj_ptr[d];
//                                 }
//                             }
//                         }
                        
//                         // 更新统计量
//                         li = li * scale_o + lij * scale_ij;
//                         mi = mi_new;
//                     }
//                 }
//             }
            
//             // ✅ 所有KV块处理完后，写入最终输出
//             for (int t = 0; t < target_seq_len; ++t) {
//                 if (li_list[t] > static_cast<T>(0)) {
//                     T inv_li = static_cast<T>(1) / li_list[t];
//                     for (int d = 0; d < head_dim; ++d) {
//                         h_o[idx_o(b, t, qh, d)] = Oi_list[t][d] * inv_li;
//                     }
//                 }
//             }
//         }
//     }
// }

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