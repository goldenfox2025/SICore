#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <mma.h>

template <typename T>
__global__ void naive_gemm_kernel(const T *A, const T *B, T *C, int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N)
        return;

    float sum = 0.0f;
    for (int k = 0; k < K; k++)
    {
        sum += static_cast<float>(A[row * K + k]) * static_cast<float>(B[col * K + k]);
    }
    C[row * N + col] = static_cast<T>(sum);
}

template <typename T, int BM = 32, int BN = 32, int BK = 32, int TM = 2, int TN = 2>
__global__ void kernel1(const T *A, const T *B, T *C, int M, int N, int K)
{
    constexpr int padding = 1;
    __shared__ T smemA[BM * (BK + padding)];
    __shared__ T smemB[BN * (BK + padding)];

    int tid = threadIdx.x;
    int global_m_base = blockIdx.x * BM;
    int global_n_base = blockIdx.y * BN;

    float acc[TM][TN] = {0.0f};

    constexpr int TID_N = BN / TN;
    int tid_m = tid / TID_N;
    int tid_n = tid % TID_N;

    for (int k_base = 0; k_base < K; k_base += BK)
    {
        constexpr int vec_size = sizeof(float4) / sizeof(T);
        const int num_threads = blockDim.x;

        for (int load_idx = tid * vec_size; load_idx < BM * BK; load_idx += num_threads * vec_size)
        {
            int smem_row = load_idx / BK;
            int smem_col = load_idx % BK;
            int global_row = global_m_base + smem_row;
            int global_col = k_base + smem_col;

            if (global_row < M && (global_col + vec_size - 1) < K)
            {

                const float4 temp_A = *reinterpret_cast<const float4 *>(&A[global_row * K + global_col]);
                for (int i = 0; i < vec_size; ++i)
                {
                    smemA[smem_row * (BK + padding) + smem_col + i] = reinterpret_cast<const T *>(&temp_A)[i];
                }
            }
            else
            {

                for (int i = 0; i < vec_size; ++i)
                {
                    if ((global_row < M) && ((global_col + i) < K))
                    {
                        smemA[smem_row * (BK + padding) + smem_col + i] = A[global_row * K + global_col + i];
                    }
                    else
                    {
                        smemA[smem_row * (BK + padding) + smem_col + i] = T(0);
                    }
                }
            }
        }

        for (int load_idx = tid * vec_size; load_idx < BN * BK; load_idx += num_threads * vec_size)
        {
            int smem_row = load_idx / BK;
            int smem_col = load_idx % BK;
            int global_row = global_n_base + smem_row;
            int global_col = k_base + smem_col;

            if (global_row < N && (global_col + vec_size - 1) < K)
            {

                const float4 temp_B = *reinterpret_cast<const float4 *>(&B[global_row * K + global_col]);
                for (int i = 0; i < vec_size; ++i)
                {
                    smemB[smem_row * (BK + padding) + smem_col + i] = reinterpret_cast<const T *>(&temp_B)[i];
                }
            }
            else
            {

                for (int i = 0; i < vec_size; ++i)
                {
                    if ((global_row < N) && ((global_col + i) < K))
                    {
                        smemB[smem_row * (BK + padding) + smem_col + i] = B[global_row * K + global_col + i];
                    }
                    else
                    {
                        smemB[smem_row * (BK + padding) + smem_col + i] = T(0);
                    }
                }
            }
        }

        __syncthreads();

        for (int k = 0; k < BK; ++k)
        {
            for (int tm = 0; tm < TM; ++tm)
            {
                for (int tn = 0; tn < TN; ++tn)
                {
                    acc[tm][tn] += static_cast<float>(smemA[(tid_m * TM + tm) * (BK + padding) + k]) * static_cast<float>(smemB[(tid_n * TN + tn) * (BK + padding) + k]);
                }
            }
        }
        __syncthreads();
    }

    // --- Write-back results ---
    for (int tm = 0; tm < TM; ++tm)
    {
        for (int tn = 0; tn < TN; ++tn)
        {
            int global_row = global_m_base + tid_m * TM + tm;
            int global_col = global_n_base + tid_n * TN + tn;
            if (global_row < M && global_col < N)
            {
                C[global_row * N + global_col] = static_cast<T>(acc[tm][tn]);
            }
        }
    }
}
template <typename T, int BM = 32, int BN = 64, int BK = 64, int WMMA_M = 16, int WMMA_N = 16, int WMMA_K = 16, int WAPR_NUM = 8>
__global__ void kernel2(const T *A, const T *B, T *C, int M, int N, int K)
{

    int warp_id = threadIdx.x / 32;
    // int lane_id = threadIdx.x % 32;
    constexpr int WARP_N_NUM = BN / WMMA_N;
    int warp_n_id = warp_id % WARP_N_NUM;
    int warp_m_id = warp_id / WARP_N_NUM;
    int global_m_base = blockIdx.x * BM;
    int global_n_base = blockIdx.y * BN;

    using FragmentA = nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, T, nvcuda::wmma::row_major>;
    using FragmentB = nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, T, nvcuda::wmma::col_major>;
    using FragmentC = nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>;

    FragmentA fragA;
    FragmentB fragB;
    FragmentC fragC;
    nvcuda::wmma::fill_fragment(fragC, float(0));
    __shared__ T smemA[BM * BK];
    __shared__ T smemB[BN * BK];
    __shared__ float smemC[BM * BN];
    constexpr int vec_size = sizeof(float4) / sizeof(T);
    for (int k_base = 0; k_base < K; k_base += BK)
    {
        for (int load_idx = threadIdx.x * vec_size; load_idx < BM * BK; load_idx += blockDim.x * vec_size)
        {
            int smem_row = load_idx / (BK);
            int smem_col = load_idx % (BK);
            int global_row = global_m_base + smem_row;
            int global_col = k_base + smem_col;
            if (global_row < M && (global_col + vec_size - 1) < K)
            {
                reinterpret_cast<float4 *>(&smemA[load_idx])[0] = reinterpret_cast<const float4 *>(&A[global_row * K + global_col])[0];
            }
            else
            {
                for (int i = 0; i < vec_size; ++i)
                {
                    if ((global_row < M) && ((global_col + i) < K))
                    {
                        smemA[load_idx + i] = A[global_row * K + global_col + i];
                    }
                    else
                    {
                        smemA[load_idx + i] = T(0);
                    }
                }
            }
        }

        for (int load_idx = threadIdx.x * vec_size; load_idx < BN * BK; load_idx += blockDim.x * vec_size)
        {
            int smem_row = load_idx / (BK);
            int smem_col = load_idx % (BK);
            int global_row = global_n_base + smem_row;
            int global_col = k_base + smem_col;
            if (global_col + vec_size - 1 < K && global_row < N)
            {
                reinterpret_cast<float4 *>(&smemB[load_idx])[0] = reinterpret_cast<const float4 *>(&B[global_row * K + global_col])[0];
            }
            else
            {
                for (int i = 0; i < vec_size; ++i)
                {
                    if (global_col + i < K && global_row < N)
                    {
                        smemB[load_idx + i] = B[global_row * K + global_col + i];
                    }
                    else
                    {
                        smemB[load_idx + i] = T(0);
                    }
                }
            }
        }

        __syncthreads();
        for (int k_step = 0; k_step < BK; k_step += WMMA_K)
        {
            const T *smemA_ptr = smemA + (warp_m_id * WMMA_M * BK) + k_step;
            const T *smemB_ptr = smemB + (warp_n_id * WMMA_N * BK) + k_step;
            nvcuda::wmma::load_matrix_sync(fragA, smemA_ptr, BK);
            nvcuda::wmma::load_matrix_sync(fragB, smemB_ptr, BK);
            nvcuda::wmma::mma_sync(fragC, fragA, fragB, fragC);
        }
        __syncthreads();
    }

    const int out_m_base = warp_m_id * WMMA_M;
    const int out_n_base = warp_n_id * WMMA_N;
    nvcuda::wmma::store_matrix_sync(&smemC[out_m_base * BN + out_n_base], fragC, BN, nvcuda::wmma::mem_row_major);
    __syncthreads();
    for (int load_idx = threadIdx.x; load_idx < BM * BN; load_idx += blockDim.x)
    {
        int smem_row = load_idx / BN;
        int smem_col = load_idx % BN;
        int global_row = global_m_base + smem_row;
        int global_col = global_n_base + smem_col;
        if (global_row < M && global_col < N)
        {
            C[global_row * N + global_col] = static_cast<T>(smemC[load_idx]);
        }
    }
}

template <typename T, int BM = 64, int BN = 64, int BK = 64, int WMMA_M = 16, int WMMA_N = 16, int WMMA_K = 16, int WAPR_NUM = 8, int K_STAGE = 2, int WARP_TILE_M = 2, int WARP_TILE_N = 1>
__global__ void kernel3(const T *A, const T *B, T *C, int M, int N, int K)
{

    int warp_id = threadIdx.x / 32;
    // int lane_id = threadIdx.x % 32;
    constexpr int WARP_N_NUM = BN / (WMMA_N * WARP_TILE_N);
    int warp_n_id = warp_id % WARP_N_NUM;
    int warp_m_id = warp_id / WARP_N_NUM;
    int global_m_base = blockIdx.x * BM;
    int global_n_base = blockIdx.y * BN;
    constexpr int SA_SIZE = BM * BK;
    constexpr int SB_SIZE = BN * BK;
    using FragmentA = nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, T, nvcuda::wmma::row_major>;
    using FragmentB = nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, T, nvcuda::wmma::col_major>;
    using FragmentC = nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>;

    FragmentC C_frag[WARP_TILE_M][WARP_TILE_N];

#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i)
    {
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j)
        {
            nvcuda::wmma::fill_fragment(C_frag[i][j], 0.0);
        }
    }

    __shared__ T smemA[K_STAGE * BM * BK];
    __shared__ T smemB[K_STAGE * BN * BK];
    __shared__ float smemC[BM * BN];
    uint32_t smem_a_base_ptr = __cvta_generic_to_shared(smemA);
    uint32_t smem_b_base_ptr = __cvta_generic_to_shared(smemB);

    constexpr int vec_size = sizeof(float4) / sizeof(T);

    // 加载除最后阶段外的数据
    for (int k_load_stage = 0; k_load_stage < (K_STAGE - 1); ++k_load_stage)
    {

        for (int load_idx = threadIdx.x * vec_size; load_idx < BM * BK; load_idx += blockDim.x * vec_size)
        {
            int smem_row = load_idx / (BK);
            int smem_col = load_idx % (BK);
            int global_row = global_m_base + smem_row;
            int global_col = k_load_stage * BK + smem_col;
            if (global_row < M && (global_col + vec_size - 1) < K)
            {
                int load_gmem_a_addr = global_row * K + global_col;
                uint32_t load_smem_a_ptr = smem_a_base_ptr + (load_idx + k_load_stage * SA_SIZE) * sizeof(T);
                CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);
            }
            else
            {
                // 当你看到这个打印，你就该知道输入前检查矩阵是否对齐要求
                printf("CP_ASYNC_CG expected to be 16 bytes aligned\n");
            }
        }
        for (int load_idx = threadIdx.x * vec_size; load_idx < BN * BK; load_idx += blockDim.x * vec_size)
        {
            int smem_row = load_idx / (BK);
            int smem_col = load_idx % (BK);
            int global_row = global_n_base + smem_row;
            int global_col = k_load_stage * BK + smem_col;
            if (global_col + vec_size - 1 < K && global_row < N)
            {
                int load_gmem_b_addr = global_row * K + global_col;
                uint32_t load_smem_b_ptr = smem_b_base_ptr + (load_idx + k_load_stage * SA_SIZE) * sizeof(T);
                CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);
            }
            else
            {
                printf("CP_ASYNC_CG expected to be 16 bytes aligned\n");
            }
        }
        CP_ASYNC_COMMIT_GROUP();
    }
    CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
    __syncthreads();

    // 这里我们要认为这个k_load是加载阶段
    // 一开始，就已经提交K_STAGE - 1 个阶段的数据
    // 所以加载从此开始
    FragmentA fragA[WARP_TILE_M];
    FragmentB fragB[WARP_TILE_N];
    for (int k_load_base = (K_STAGE - 1) * BK; k_load_base < K; k_load_base += BK)
    {
        // 以K_STAGE = 2为例
        // 一开始，意味着已经加载了s1阶段的数据
        // k_load = BK
        const int k_load_stage = k_load_base / BK;
        int smem_sel = (k_load_stage + 1) % K_STAGE;
        int smem_sel_next = k_load_stage % K_STAGE;

        for (int load_idx = threadIdx.x * vec_size; load_idx < BM * BK; load_idx += blockDim.x * vec_size)
        {
            int smem_row = load_idx / (BK);
            int smem_col = load_idx % (BK);
            int global_row = global_m_base + smem_row;
            int global_col = k_load_base + smem_col;
            if (global_row < M && (global_col + vec_size - 1) < K)
            {
                int load_gmem_a_addr = global_row * K + global_col;
                uint32_t load_smem_a_ptr = smem_a_base_ptr + (load_idx + smem_sel_next * SA_SIZE) * sizeof(T);
                CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);
            }
            else
            {
                printf("CP_ASYNC_CG expected to be 16 bytes aligned\n");
            }
        }
        for (int load_idx = threadIdx.x * vec_size; load_idx < BN * BK; load_idx += blockDim.x * vec_size)
        {
            int smem_row = load_idx / (BK);
            int smem_col = load_idx % (BK);
            int global_row = global_n_base + smem_row;
            int global_col = k_load_base + smem_col;
            if (global_col + vec_size - 1 < K && global_row < N)
            {
                int load_gmem_b_addr = global_row * K + global_col;
                uint32_t load_smem_b_ptr = smem_b_base_ptr + (load_idx + smem_sel_next * SB_SIZE) * sizeof(T);
                CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);
            }
            else
            {
                printf("CP_ASYNC_CG expected to be 16 bytes aligned\n");
            }
        }
        CP_ASYNC_COMMIT_GROUP();

        for (int TILE_K = 0; TILE_K < BK; TILE_K += WMMA_K)
        {
            for (int i = 0; i < WARP_TILE_M; ++i)
            {
                int warp_smem_a_m = warp_m_id * WMMA_M * WARP_TILE_M + i * WMMA_M;
                int warp_smem_a_k = TILE_K;
                T *warp_smem_a_ptr = smemA + warp_smem_a_m * BK + warp_smem_a_k + smem_sel * SA_SIZE;
                nvcuda::wmma::load_matrix_sync(fragA[i], warp_smem_a_ptr, BK);
            }
            for (int i = 0; i < WARP_TILE_N; ++i)
            {
                int warp_smem_b_n = warp_n_id * WMMA_N * WARP_TILE_N + i * WMMA_N;
                int warp_smem_b_k = TILE_K;
                T *warp_smem_b_ptr = smemB + warp_smem_b_n * BK + warp_smem_b_k + smem_sel * SB_SIZE;
                nvcuda::wmma::load_matrix_sync(fragB[i], warp_smem_b_ptr, BK);
            }
            for (int i = 0; i < WARP_TILE_M; ++i)
            {
                for (int j = 0; j < WARP_TILE_N; ++j)
                {
                    nvcuda::wmma::mma_sync(C_frag[i][j], fragA[i], fragB[j], C_frag[i][j]);
                }
            }
        }

        CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
        __syncthreads();
    }

    // 主循环结束
    if ((K_STAGE - 2) > 0)
    {
        CP_ASYNC_WAIT_GROUP(0);
        __syncthreads();
    }

    // 计算剩余阶段
    for (int k_load = 0; k_load < K_STAGE - 1; ++k_load)
    {

        // K/BK 是总共的阶段数，减去K_STAGE - 1
        const int stage_sel = ((K / BK - (K_STAGE - 1) + k_load) % K_STAGE);
        for (int TILE_K = 0; TILE_K < BK; TILE_K += WMMA_K)
        {
            for (int i = 0; i < WARP_TILE_M; ++i)
            {
                int warp_smem_a_m = warp_m_id * WMMA_M * WARP_TILE_M + i * WMMA_M;
                int warp_smem_a_k = TILE_K;
                T *warp_smem_a_ptr = smemA + warp_smem_a_m * BK + warp_smem_a_k + stage_sel * SA_SIZE;
                nvcuda::wmma::load_matrix_sync(fragA[i], warp_smem_a_ptr, BK);
            }
            for (int i = 0; i < WARP_TILE_N; ++i)
            {
                int warp_smem_b_n = warp_n_id * WMMA_N * WARP_TILE_N + i * WMMA_N;
                int warp_smem_b_k = TILE_K;
                T *warp_smem_b_ptr = smemB + warp_smem_b_n * BK + warp_smem_b_k + stage_sel * SB_SIZE;
                nvcuda::wmma::load_matrix_sync(fragB[i], warp_smem_b_ptr, BK);
            }
            for (int i = 0; i < WARP_TILE_M; ++i)
            {
                for (int j = 0; j < WARP_TILE_N; ++j)
                {
                    nvcuda::wmma::mma_sync(C_frag[i][j], fragA[i], fragB[j], C_frag[i][j]);
                }
            }
        }
    }

    // 写回smemC
    for (int i = 0; i < WARP_TILE_M; ++i)
    {
        for (int j = 0; j < WARP_TILE_N; ++j)
        {
            const int warp_smem_c_m = warp_m_id * WMMA_M * WARP_TILE_M + i * WMMA_M;
            const int warp_smem_c_n = warp_n_id * WMMA_N * WARP_TILE_N + j * WMMA_N;

            nvcuda::wmma::store_matrix_sync(&smemC[warp_smem_c_m * BN + warp_smem_c_n], C_frag[i][j], BN, nvcuda::wmma::mem_row_major);
        }
    }
    __syncthreads();

    for (int load_idx = threadIdx.x; load_idx < BM * BN; load_idx += blockDim.x)
    {
        int smem_row = load_idx / BN;
        int smem_col = load_idx % BN;
        int global_row = global_m_base + smem_row;
        int global_col = global_n_base + smem_col;
        if (global_row < M && global_col < N)
        {
            C[global_row * N + global_col] = static_cast<T>(smemC[load_idx]);
        }
    }
}

// 直接基于kernel3，尝试使用mma替换掉wmma，方便进一步优化
// 只针对bf16
template <typename T, int BM, int BN, int BK, int WMMA_M, int WMMA_N, int WMMA_K, int WAPR_NUM, int K_STAGE, int WARP_TILE_M, int WARP_TILE_N>
__global__ void kernel4(const T *A, const T *B, T *C, int M, int N, int K)
{

    int warp_id = threadIdx.x / 32;
    // int lane_id = threadIdx.x % 32;
    constexpr int WARP_N_NUM = BN / (WMMA_N * WARP_TILE_N);
    int warp_n_id = warp_id % WARP_N_NUM;
    int warp_m_id = warp_id / WARP_N_NUM;
    int global_m_base = blockIdx.x * BM;
    int global_n_base = blockIdx.y * BN;
    constexpr int SA_SIZE = BM * BK;
    constexpr int SB_SIZE = BN * BK;
    const int lane_id = threadIdx.x % 32;

    __shared__ T smemA[K_STAGE * BM * BK];
    __shared__ T smemB[K_STAGE * BN * BK];
    // __shared__ float smemC[BM * BN];
    uint32_t smem_a_base_ptr = __cvta_generic_to_shared(smemA);
    uint32_t smem_b_base_ptr = __cvta_generic_to_shared(smemB);

    constexpr int vec_size = sizeof(float4) / sizeof(T);

    uint32_t RC[WARP_TILE_M][WARP_TILE_N][4];
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i)
    {
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j)
        {
            RC[i][j][0] = 0;
            RC[i][j][1] = 0;
            RC[i][j][2] = 0;
            RC[i][j][3] = 0;
        }
    }

    // 加载除最后阶段外的数据
    for (int k_load_stage = 0; k_load_stage < (K_STAGE - 1); ++k_load_stage)
    {

        for (int load_idx = threadIdx.x * vec_size; load_idx < BM * BK; load_idx += blockDim.x * vec_size)
        {
            int smem_row = load_idx / (BK);
            int smem_col = load_idx % (BK);
            int global_row = global_m_base + smem_row;
            int global_col = k_load_stage * BK + smem_col;
            if (global_row < M && (global_col + vec_size - 1) < K)
            {
                int load_gmem_a_addr = global_row * K + global_col;
                uint32_t load_smem_a_ptr = smem_a_base_ptr + (load_idx + k_load_stage * SA_SIZE) * sizeof(T);
                CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);
            }
            else
            {
                printf("CP_ASYNC_CG expected to be 16 bytes aligned\n");
            }
        }
        for (int load_idx = threadIdx.x * vec_size; load_idx < BN * BK; load_idx += blockDim.x * vec_size)
        {
            int smem_row = load_idx / (BK);
            int smem_col = load_idx % (BK);
            int global_row = global_n_base + smem_row;
            int global_col = k_load_stage * BK + smem_col;
            if (global_col + vec_size - 1 < K && global_row < N)
            {
                int load_gmem_b_addr = global_row * K + global_col;
                uint32_t load_smem_b_ptr = smem_b_base_ptr + (load_idx + k_load_stage * SB_SIZE) * sizeof(T);
                CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);
            }
            else
            {
                printf("CP_ASYNC_CG expected to be 16 bytes aligned\n");
            }
        }
        CP_ASYNC_COMMIT_GROUP();
    }
    CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
    __syncthreads();

    // 这里我们要认为这个k_load是加载阶段
    // 一开始，就已经提交K_STAGE - 1 个阶段的数据
    // 所以加载从此开始

    // 对于不同的mma指令
    // 和wmma不支持小于16 16 16的形状不同
    // mma可以灵活处理16 8 16这样的形状
    // 下面是用于储存当前线程读取结果的
    uint32_t RA[WARP_TILE_M][4];
    uint32_t RB[WARP_TILE_N][2];
    for (int k_load_base = (K_STAGE - 1) * BK; k_load_base < K; k_load_base += BK)
    {
        // 以K_STAGE = 2为例
        // 一开始，意味着已经加载了s1阶段的数据
        // k_load = BK
        const int k_load_stage = k_load_base / BK;
        int smem_sel = (k_load_stage + 1) % K_STAGE;
        int smem_sel_next = k_load_stage % K_STAGE;

        for (int load_idx = threadIdx.x * vec_size; load_idx < BM * BK; load_idx += blockDim.x * vec_size)
        {
            int smem_row = load_idx / (BK);
            int smem_col = load_idx % (BK);
            int global_row = global_m_base + smem_row;
            int global_col = k_load_base + smem_col;
            if (global_row < M && (global_col + vec_size - 1) < K)
            {
                int load_gmem_a_addr = global_row * K + global_col;
                uint32_t load_smem_a_ptr = smem_a_base_ptr + (load_idx + smem_sel_next * SA_SIZE) * sizeof(T);
                CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);
            }
            else
            {
                printf("CP_ASYNC_CG expected to be 16 bytes aligned\n");
            }
        }
        for (int load_idx = threadIdx.x * vec_size; load_idx < BN * BK; load_idx += blockDim.x * vec_size)
        {
            int smem_row = load_idx / (BK);
            int smem_col = load_idx % (BK);
            int global_row = global_n_base + smem_row;
            int global_col = k_load_base + smem_col;
            if (global_col + vec_size - 1 < K && global_row < N)
            {
                int load_gmem_b_addr = global_row * K + global_col;
                uint32_t load_smem_b_ptr = smem_b_base_ptr + (load_idx + smem_sel_next * SB_SIZE) * sizeof(T);
                CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);
            }
            else
            {
                printf("CP_ASYNC_CG expected to be 16 bytes aligned\n");
            }
        }
        CP_ASYNC_COMMIT_GROUP();

        for (int TILE_K = 0; TILE_K < BK; TILE_K += WMMA_K)
        {
            for (int i = 0; i < WARP_TILE_M; ++i)
            {
                int warp_smem_a_m = warp_m_id * WMMA_M * WARP_TILE_M + i * WMMA_M;
                int warp_smem_a_k = TILE_K;
                // 16 * 16 矩阵的起点
                T *warp_smem_a_ptr = smemA + warp_smem_a_m * BK + warp_smem_a_k + smem_sel * SA_SIZE;
                T *lane_smem_a_ptr = warp_smem_a_ptr + (lane_id % 16) * BK + (lane_id / 16) * vec_size;
                uint32_t ptr = __cvta_generic_to_shared(lane_smem_a_ptr);
                LDMATRIX_X4(RA[i][0], RA[i][1], RA[i][2], RA[i][3], ptr);
            }
            for (int i = 0; i < WARP_TILE_N; ++i)
            {
                int warp_smem_b_n = warp_n_id * WMMA_N * WARP_TILE_N + i * WMMA_N;
                int warp_smem_b_k = TILE_K;
                T *warp_smem_b_ptr = smemB + warp_smem_b_n * BK + warp_smem_b_k + smem_sel * SB_SIZE;
                T *lane_smem_b_ptr = warp_smem_b_ptr + (lane_id % 8) * BK + (lane_id / 8) * vec_size;
                uint32_t ptr = __cvta_generic_to_shared(lane_smem_b_ptr);
                LDMATRIX_X2(RB[i][0], RB[i][1], ptr);
            }
            for (int i = 0; i < WARP_TILE_M; ++i)
            {
                for (int j = 0; j < WARP_TILE_N; ++j)
                {
                    MMA16816_BF16(RC[i][j][0], RC[i][j][1], RC[i][j][2], RC[i][j][3], RA[i][0], RA[i][1], RA[i][2], RA[i][3], RB[j][0], RB[j][1], RC[i][j][0], RC[i][j][1], RC[i][j][2], RC[i][j][3]);
                }
            }
        }

        CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
        __syncthreads();
    }

    // 主循环结束
    if ((K_STAGE - 2) > 0)
    {
        CP_ASYNC_WAIT_GROUP(0);
        __syncthreads();
    }

    // 计算剩余阶段
    for (int k_load = 0; k_load < K_STAGE - 1; ++k_load)
    {

        // K/BK 是总共的阶段数，减去K_STAGE - 1
        const int stage_sel = ((K / BK - (K_STAGE - 1) + k_load) % K_STAGE);
        for (int TILE_K = 0; TILE_K < BK; TILE_K += WMMA_K)
        {
            for (int i = 0; i < WARP_TILE_M; ++i)
            {
                int warp_smem_a_m = warp_m_id * WMMA_M * WARP_TILE_M + i * WMMA_M;
                int warp_smem_a_k = TILE_K;
                T *warp_smem_a_ptr = smemA + warp_smem_a_m * BK + warp_smem_a_k + stage_sel * SA_SIZE;
                T *lane_smem_a_ptr = warp_smem_a_ptr + (lane_id % 16) * BK + (lane_id / 16) * vec_size;

                uint32_t ptr = __cvta_generic_to_shared(lane_smem_a_ptr);
                LDMATRIX_X4(RA[i][0], RA[i][1], RA[i][2], RA[i][3], ptr);
            }
            for (int i = 0; i < WARP_TILE_N; ++i)
            {
                int warp_smem_b_n = warp_n_id * WMMA_N * WARP_TILE_N + i * WMMA_N;
                int warp_smem_b_k = TILE_K;
                T *warp_smem_b_ptr = smemB + warp_smem_b_n * BK + warp_smem_b_k + stage_sel * SB_SIZE;
                T *lane_smem_b_ptr = warp_smem_b_ptr + (lane_id % 8) * BK + (lane_id / 8) * vec_size;

                uint32_t ptr = __cvta_generic_to_shared(lane_smem_b_ptr);
                LDMATRIX_X2(RB[i][0], RB[i][1], ptr);
            }
            for (int i = 0; i < WARP_TILE_M; ++i)
            {
                for (int j = 0; j < WARP_TILE_N; ++j)
                {
                    MMA16816_BF16(RC[i][j][0], RC[i][j][1], RC[i][j][2], RC[i][j][3], RA[i][0], RA[i][1], RA[i][2], RA[i][3], RB[j][0], RB[j][1], RC[i][j][0], RC[i][j][1], RC[i][j][2], RC[i][j][3]);
                }
            }
        }
    }

    // 接下来是最重要的部分
    // 将 RC 的 4×fp32 写回到输出矩阵 C
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i)
    {
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j)
        {

            // 1. 本 16×8 子块在整张 C 里的左上角坐标
            const int tile_m0 = global_m_base + (warp_m_id * WMMA_M * WARP_TILE_M) + i * WMMA_M;
            const int tile_n0 = global_n_base + (warp_n_id * WMMA_N * WARP_TILE_N) + j * WMMA_N;

            // 2. 计算当前 lane 在该子块内负责的 2×2 行列偏移
            int group = lane_id >> 2; // 0..7  → 行基
            int tid4 = lane_id & 3;   // 0..3  → 列基
            int row0 = group;         // 上半行
            int row1 = group + 8;     // 下半行
            int col0 = 2 * tid4;      // 左列
            int col1 = 2 * tid4 + 1;  // 右列

            // 3. 取出 4 个累加结果（FP32） —— 注意 RC 已经是 uint32_t
            float v0 = __uint_as_float(RC[i][j][0]); // 对应 (row0 , col0)
            float v1 = __uint_as_float(RC[i][j][1]); // 对应 (row0 , col1)
            float v2 = __uint_as_float(RC[i][j][2]); // 对应 (row1 , col0)
            float v3 = __uint_as_float(RC[i][j][3]); // 对应 (row1 , col1)

            // 4. 折算成全局内存下标并越界保护（必要时可去掉保护以提速）
            int gidx0 = (tile_m0 + row0) * N + (tile_n0 + col0);
            int gidx1 = (tile_m0 + row0) * N + (tile_n0 + col1);
            int gidx2 = (tile_m0 + row1) * N + (tile_n0 + col0);
            int gidx3 = (tile_m0 + row1) * N + (tile_n0 + col1);

            if ((tile_m0 + row0) < M && (tile_n0 + col0) < N)
                C[gidx0] = v0;
            if ((tile_m0 + row0) < M && (tile_n0 + col1) < N)
                C[gidx1] = v1;
            if ((tile_m0 + row1) < M && (tile_n0 + col0) < N)
                C[gidx2] = v2;
            if ((tile_m0 + row1) < M && (tile_n0 + col1) < N)
                C[gidx3] = v3;
        }
    }
}

// Swizzle helper functions
template <const int kColStride = 16, const int kStep = 8>
static __device__ __forceinline__ int swizzle_permuted_j(int i, int j) {
  // for col_stride > 16, we have to permute it using col major ZigZag order.
  // e.g, A smem logical layout [Br,d]=[Br,64] -> store layout [4][Br][16].
  static_assert(kColStride <= 16, "kColStride must <= 16");
  // swizzle: ((int(j / kStep) ^ int(i / 4)) % int(kColStride / kStep)) * kStep;
  static_assert(kStep == 4 || kStep == 8, "kStep must be 8 or 4.");
  static_assert(kColStride % kStep == 0,
                "kColStride must be multiple of kStep.");
  if constexpr (kStep == 8) {
    return (((j >> 3) ^ (i >> 2)) % (kColStride >> 3)) << 3;
  } else {
    static_assert(kStep == 4);
    return (((j >> 2) ^ (i >> 2)) % (kColStride >> 2)) << 2;
  }
}

// i: row index; j: col index
template <const int kMmaAtomK = 16>
static __device__ __forceinline__ int swizzle_permuted_A_j(int i, int j) {
  return swizzle_permuted_j<kMmaAtomK, 8>(i, j);
}

// i: row index; j: col index
template <const int kMmaAtomK = 16>
static __device__ __forceinline__ int swizzle_permuted_B_j(int i, int j) {
  return swizzle_permuted_j<kMmaAtomK, 8>(i, j);
}

template <typename T, int BM, int BN, int BK, int WMMA_M, int WMMA_N, int WMMA_K, int WAPR_NUM, int K_STAGE, int WARP_TILE_M, int WARP_TILE_N>
__global__ void kernel5(const T *A, const T *B, T *C, int M, int N, int K)
{
    int warp_id = threadIdx.x / 32;
    // int lane_id = threadIdx.x % 32;
    constexpr int WARP_N_NUM = BN / (WMMA_N * WARP_TILE_N);
    int warp_n_id = warp_id % WARP_N_NUM;
    int warp_m_id = warp_id / WARP_N_NUM;
    int global_m_base = blockIdx.x * BM;
    int global_n_base = blockIdx.y * BN;
    constexpr int SA_SIZE = BM * BK;
    constexpr int SB_SIZE = BN * BK;
    const int lane_id = threadIdx.x % 32;

    __shared__ T smemA[K_STAGE * BM * BK];
    __shared__ T smemB[K_STAGE * BN * BK];
    // __shared__ float smemC[BM * BN];
    uint32_t smem_a_base_ptr = __cvta_generic_to_shared(smemA);
    uint32_t smem_b_base_ptr = __cvta_generic_to_shared(smemB);

    constexpr int vec_size = sizeof(float4) / sizeof(T);

    uint32_t RC[WARP_TILE_M][WARP_TILE_N][4];
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i)
    {
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j)
        {
            RC[i][j][0] = 0;
            RC[i][j][1] = 0;
            RC[i][j][2] = 0;
            RC[i][j][3] = 0;
        }
    }

    // 加载除最后阶段外的数据 - 应用swizzle
    for (int k_load_stage = 0; k_load_stage < (K_STAGE - 1); ++k_load_stage)
    {
        // 加载A矩阵数据 - 应用swizzle
        for (int load_idx = threadIdx.x * vec_size; load_idx < BM * BK; load_idx += blockDim.x * vec_size)
        {
            int smem_row = load_idx / (BK);
            int smem_col = load_idx % (BK);
            int global_row = global_m_base + smem_row;
            int global_col = k_load_stage * BK + smem_col;
            
            if (global_row < M && (global_col + vec_size - 1) < K)
            {
                int load_gmem_a_addr = global_row * K + global_col;
                // 应用swizzle计算存储位置
                int swizzled_col = swizzle_permuted_A_j(smem_row, smem_col);
                int swizzled_idx = smem_row * BK + swizzled_col;
                uint32_t load_smem_a_ptr = smem_a_base_ptr + (swizzled_idx + k_load_stage * SA_SIZE) * sizeof(T);
                CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);
            }
            else
            {
                printf("CP_ASYNC_CG expected to be 16 bytes aligned\n");
            }
        }
        
        // 加载B矩阵数据 - 应用swizzle
        for (int load_idx = threadIdx.x * vec_size; load_idx < BN * BK; load_idx += blockDim.x * vec_size)
        {
            int smem_row = load_idx / (BK);
            int smem_col = load_idx % (BK);
            int global_row = global_n_base + smem_row;
            int global_col = k_load_stage * BK + smem_col;
            
            if (global_col + vec_size - 1 < K && global_row < N)
            {
                int load_gmem_b_addr = global_row * K + global_col;
                // 应用swizzle计算存储位置
                int swizzled_col = swizzle_permuted_B_j(smem_row, smem_col);
                int swizzled_idx = smem_row * BK + swizzled_col;
                uint32_t load_smem_b_ptr = smem_b_base_ptr + (swizzled_idx + k_load_stage * SB_SIZE) * sizeof(T);
                CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);
            }
            else
            {
                printf("CP_ASYNC_CG expected to be 16 bytes aligned\n");
            }
        }
        CP_ASYNC_COMMIT_GROUP();
    }
    CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
    __syncthreads();

    // 主循环 - 应用swizzle
    uint32_t RA[WARP_TILE_M][4];
    uint32_t RB[WARP_TILE_N][2];
    for (int k_load_base = (K_STAGE - 1) * BK; k_load_base < K; k_load_base += BK)
    {
        const int k_load_stage = k_load_base / BK;
        int smem_sel = (k_load_stage + 1) % K_STAGE;
        int smem_sel_next = k_load_stage % K_STAGE;

        // 加载A矩阵数据 - 应用swizzle
        for (int load_idx = threadIdx.x * vec_size; load_idx < BM * BK; load_idx += blockDim.x * vec_size)
        {
            int smem_row = load_idx / (BK);
            int smem_col = load_idx % (BK);
            int global_row = global_m_base + smem_row;
            int global_col = k_load_base + smem_col;
            
            if (global_row < M && (global_col + vec_size - 1) < K)
            {
                int load_gmem_a_addr = global_row * K + global_col;
                // 应用swizzle计算存储位置
                int swizzled_col = swizzle_permuted_A_j(smem_row, smem_col);
                int swizzled_idx = smem_row * BK + swizzled_col;
                uint32_t load_smem_a_ptr = smem_a_base_ptr + (swizzled_idx + smem_sel_next * SA_SIZE) * sizeof(T);
                CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);
            }
            else
            {
                printf("CP_ASYNC_CG expected to be 16 bytes aligned\n");
            }
        }
        
        // 加载B矩阵数据 - 应用swizzle
        for (int load_idx = threadIdx.x * vec_size; load_idx < BN * BK; load_idx += blockDim.x * vec_size)
        {
            int smem_row = load_idx / (BK);
            int smem_col = load_idx % (BK);
            int global_row = global_n_base + smem_row;
            int global_col = k_load_base + smem_col;
            
            if (global_col + vec_size - 1 < K && global_row < N)
            {
                int load_gmem_b_addr = global_row * K + global_col;
                // 应用swizzle计算存储位置
                int swizzled_col = swizzle_permuted_B_j(smem_row, smem_col);
                int swizzled_idx = smem_row * BK + swizzled_col;
                uint32_t load_smem_b_ptr = smem_b_base_ptr + (swizzled_idx + smem_sel_next * SB_SIZE) * sizeof(T);
                CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);
            }
            else
            {
                printf("CP_ASYNC_CG expected to be 16 bytes aligned\n");
            }
        }
        CP_ASYNC_COMMIT_GROUP();

        // 计算部分 - 读取时需要应用swizzle
        for (int TILE_K = 0; TILE_K < BK; TILE_K += WMMA_K)
        {
            // 读取A矩阵数据 - 应用swizzle
            for (int i = 0; i < WARP_TILE_M; ++i)
            {
                int warp_smem_a_m = warp_m_id * WMMA_M * WARP_TILE_M + i * WMMA_M;
                int warp_smem_a_k = TILE_K;
                
                // 计算原始位置
                int base_row = warp_smem_a_m + (lane_id % 16);
                int base_col = warp_smem_a_k + (lane_id / 16) * vec_size;
                
                // 应用swizzle计算实际存储位置
                int swizzled_col = swizzle_permuted_A_j(base_row, base_col);
                T *lane_smem_a_ptr = smemA + base_row * BK + swizzled_col + smem_sel * SA_SIZE;
                
                uint32_t ptr = __cvta_generic_to_shared(lane_smem_a_ptr);
                LDMATRIX_X4(RA[i][0], RA[i][1], RA[i][2], RA[i][3], ptr);
            }
            
            // 读取B矩阵数据 - 应用swizzle
            for (int i = 0; i < WARP_TILE_N; ++i)
            {
                int warp_smem_b_n = warp_n_id * WMMA_N * WARP_TILE_N + i * WMMA_N;
                int warp_smem_b_k = TILE_K;
                
                // 计算原始位置
                int base_row = warp_smem_b_n + (lane_id % 8);
                int base_col = warp_smem_b_k + (lane_id / 8) * vec_size;
                
                // 应用swizzle计算实际存储位置
                int swizzled_col = swizzle_permuted_B_j(base_row, base_col);
                T *lane_smem_b_ptr = smemB + base_row * BK + swizzled_col + smem_sel * SB_SIZE;
                
                uint32_t ptr = __cvta_generic_to_shared(lane_smem_b_ptr);
                LDMATRIX_X2(RB[i][0], RB[i][1], ptr);
            }
            
            // 执行矩阵乘法
            for (int i = 0; i < WARP_TILE_M; ++i)
            {
                for (int j = 0; j < WARP_TILE_N; ++j)
                {
                    MMA16816_BF16(RC[i][j][0], RC[i][j][1], RC[i][j][2], RC[i][j][3], 
                                  RA[i][0], RA[i][1], RA[i][2], RA[i][3], 
                                  RB[j][0], RB[j][1], 
                                  RC[i][j][0], RC[i][j][1], RC[i][j][2], RC[i][j][3]);
                }
            }
        }

        CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
        __syncthreads();
    }

    // 主循环结束
    if ((K_STAGE - 2) > 0)
    {
        CP_ASYNC_WAIT_GROUP(0);
        __syncthreads();
    }

    // 计算剩余阶段 - 应用swizzle
    for (int k_load = 0; k_load < K_STAGE - 1; ++k_load)
    {
        const int stage_sel = ((K / BK - (K_STAGE - 1) + k_load) % K_STAGE);
        for (int TILE_K = 0; TILE_K < BK; TILE_K += WMMA_K)
        {
            // 读取A矩阵数据 - 应用swizzle
            for (int i = 0; i < WARP_TILE_M; ++i)
            {
                int warp_smem_a_m = warp_m_id * WMMA_M * WARP_TILE_M + i * WMMA_M;
                int warp_smem_a_k = TILE_K;
                
                // 计算原始位置
                int base_row = warp_smem_a_m + (lane_id % 16);
                int base_col = warp_smem_a_k + (lane_id / 16) * vec_size;
                
                // 应用swizzle计算实际存储位置
                int swizzled_col = swizzle_permuted_A_j(base_row, base_col);
                T *lane_smem_a_ptr = smemA + base_row * BK + swizzled_col + stage_sel * SA_SIZE;

                uint32_t ptr = __cvta_generic_to_shared(lane_smem_a_ptr);
                LDMATRIX_X4(RA[i][0], RA[i][1], RA[i][2], RA[i][3], ptr);
            }
            
            // 读取B矩阵数据 - 应用swizzle
            for (int i = 0; i < WARP_TILE_N; ++i)
            {
                int warp_smem_b_n = warp_n_id * WMMA_N * WARP_TILE_N + i * WMMA_N;
                int warp_smem_b_k = TILE_K;
                
                // 计算原始位置
                int base_row = warp_smem_b_n + (lane_id % 8);
                int base_col = warp_smem_b_k + (lane_id / 8) * vec_size;
                
                // 应用swizzle计算实际存储位置
                int swizzled_col = swizzle_permuted_B_j(base_row, base_col);
                T *lane_smem_b_ptr = smemB + base_row * BK + swizzled_col + stage_sel * SB_SIZE;

                uint32_t ptr = __cvta_generic_to_shared(lane_smem_b_ptr);
                LDMATRIX_X2(RB[i][0], RB[i][1], ptr);
            }
            
            // 执行矩阵乘法
            for (int i = 0; i < WARP_TILE_M; ++i)
            {
                for (int j = 0; j < WARP_TILE_N; ++j)
                {
                    MMA16816_BF16(RC[i][j][0], RC[i][j][1], RC[i][j][2], RC[i][j][3], 
                                  RA[i][0], RA[i][1], RA[i][2], RA[i][3], 
                                  RB[j][0], RB[j][1], 
                                  RC[i][j][0], RC[i][j][1], RC[i][j][2], RC[i][j][3]);
                }
            }
        }
    }

    // 将RC的4×fp32写回到输出矩阵C
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i)
    {
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j)
        {
            // 1. 本16×8子块在整张C里的左上角坐标
            const int tile_m0 = global_m_base + (warp_m_id * WMMA_M * WARP_TILE_M) + i * WMMA_M;
            const int tile_n0 = global_n_base + (warp_n_id * WMMA_N * WARP_TILE_N) + j * WMMA_N;

            // 2. 计算当前lane在该子块内负责的2×2行列偏移
            int group = lane_id >> 2; // 0..7  → 行基
            int tid4 = lane_id & 3;   // 0..3  → 列基
            int row0 = group;         // 上半行
            int row1 = group + 8;     // 下半行
            int col0 = 2 * tid4;      // 左列
            int col1 = 2 * tid4 + 1;  // 右列

            // 3. 取出4个累加结果（FP32）
            float v0 = __uint_as_float(RC[i][j][0]); // 对应 (row0, col0)
            float v1 = __uint_as_float(RC[i][j][1]); // 对应 (row0, col1)
            float v2 = __uint_as_float(RC[i][j][2]); // 对应 (row1, col0)
            float v3 = __uint_as_float(RC[i][j][3]); // 对应 (row1, col1)

            // 4. 折算成全局内存下标并越界保护
            int gidx0 = (tile_m0 + row0) * N + (tile_n0 + col0);
            int gidx1 = (tile_m0 + row0) * N + (tile_n0 + col1);
            int gidx2 = (tile_m0 + row1) * N + (tile_n0 + col0);
            int gidx3 = (tile_m0 + row1) * N + (tile_n0 + col1);

            if ((tile_m0 + row0) < M && (tile_n0 + col0) < N)
                C[gidx0] = v0;
            if ((tile_m0 + row0) < M && (tile_n0 + col1) < N)
                C[gidx1] = v1;
            if ((tile_m0 + row1) < M && (tile_n0 + col0) < N)
                C[gidx2] = v2;
            if ((tile_m0 + row1) < M && (tile_n0 + col1) < N)
                C[gidx3] = v3;
        }
    }
}