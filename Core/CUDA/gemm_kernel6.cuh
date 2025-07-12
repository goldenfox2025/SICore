#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "../../Include/ptx_common.h"

// kernel6: Based on kernel5 but using 16x16x16 MMA dimensions
template <typename T, int BM, int BN, int BK, int WMMA_M, int WMMA_N, int WMMA_K, int WAPR_NUM, int K_STAGE, int WARP_TILE_M, int WARP_TILE_N>
__global__ void kernel6(const T *A, const T *B, T *C, int M, int N, int K)
{
    int warp_id = threadIdx.x / 32;
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
    uint32_t smem_a_base_ptr = __cvta_generic_to_shared(smemA);
    uint32_t smem_b_base_ptr = __cvta_generic_to_shared(smemB);

    constexpr int vec_size = sizeof(float4) / sizeof(T);

    // For 16x16x16, we have 8 output registers per tile
    uint32_t RC[WARP_TILE_M][WARP_TILE_N][8];
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i)
    {
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j)
        {
            RC[i][j][0] = 0; RC[i][j][1] = 0; RC[i][j][2] = 0; RC[i][j][3] = 0;
            RC[i][j][4] = 0; RC[i][j][5] = 0; RC[i][j][6] = 0; RC[i][j][7] = 0;
        }
    }

    // Load initial stages - apply swizzle
    for (int k_load_stage = 0; k_load_stage < (K_STAGE - 1); ++k_load_stage)
    {
        // Load A matrix data - apply swizzle
        for (int load_idx = threadIdx.x * vec_size; load_idx < BM * BK; load_idx += blockDim.x * vec_size)
        {
            int smem_row = load_idx / (BK);
            int smem_col = load_idx % (BK);
            int global_row = global_m_base + smem_row;
            int global_col = k_load_stage * BK + smem_col;
            
            if (global_row < M && (global_col + vec_size - 1) < K)
            {
                int load_gmem_a_addr = global_row * K + global_col;
                // Apply swizzle to calculate storage position
                int swizzled_col = swizzle_permuted_A_j(smem_row, smem_col);
                int swizzled_idx = smem_row * BK + swizzled_col;
                uint32_t load_smem_a_ptr = smem_a_base_ptr + (swizzled_idx + k_load_stage * SA_SIZE) * sizeof(T);
                CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);
            }
        }
        
        // Load B matrix data - apply swizzle
        for (int load_idx = threadIdx.x * vec_size; load_idx < BN * BK; load_idx += blockDim.x * vec_size)
        {
            int smem_row = load_idx / (BK);
            int smem_col = load_idx % (BK);
            int global_row = global_n_base + smem_row;
            int global_col = k_load_stage * BK + smem_col;
            
            if (global_col + vec_size - 1 < K && global_row < N)
            {
                int load_gmem_b_addr = global_row * K + global_col;
                // Apply swizzle to calculate storage position
                int swizzled_col = swizzle_permuted_B_j(smem_row, smem_col);
                int swizzled_idx = smem_row * BK + swizzled_col;
                uint32_t load_smem_b_ptr = smem_b_base_ptr + (swizzled_idx + k_load_stage * SB_SIZE) * sizeof(T);
                CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);
            }
        }
        CP_ASYNC_COMMIT_GROUP();
    }
    CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
    __syncthreads();

    // Main loop - apply swizzle
    uint32_t RA[WARP_TILE_M][4];  // A矩阵保持4个寄存器
    uint32_t RB[WARP_TILE_N][4];  // B矩阵需要4个寄存器来支持16x16（两个8的组合）
    for (int k_load_base = (K_STAGE - 1) * BK; k_load_base < K; k_load_base += BK)
    {
        const int k_load_stage = k_load_base / BK;
        int smem_sel = (k_load_stage + 1) % K_STAGE;
        int smem_sel_next = k_load_stage % K_STAGE;

        // Load A matrix data - apply swizzle
        for (int load_idx = threadIdx.x * vec_size; load_idx < BM * BK; load_idx += blockDim.x * vec_size)
        {
            int smem_row = load_idx / (BK);
            int smem_col = load_idx % (BK);
            int global_row = global_m_base + smem_row;
            int global_col = k_load_base + smem_col;
            
            if (global_row < M && (global_col + vec_size - 1) < K)
            {
                int load_gmem_a_addr = global_row * K + global_col;
                // Apply swizzle to calculate storage position
                int swizzled_col = swizzle_permuted_A_j(smem_row, smem_col);
                int swizzled_idx = smem_row * BK + swizzled_col;
                uint32_t load_smem_a_ptr = smem_a_base_ptr + (swizzled_idx + smem_sel_next * SA_SIZE) * sizeof(T);
                CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);
            }
        }
        
        // Load B matrix data - apply swizzle
        for (int load_idx = threadIdx.x * vec_size; load_idx < BN * BK; load_idx += blockDim.x * vec_size)
        {
            int smem_row = load_idx / (BK);
            int smem_col = load_idx % (BK);
            int global_row = global_n_base + smem_row;
            int global_col = k_load_base + smem_col;
            
            if (global_col + vec_size - 1 < K && global_row < N)
            {
                int load_gmem_b_addr = global_row * K + global_col;
                // Apply swizzle to calculate storage position
                int swizzled_col = swizzle_permuted_B_j(smem_row, smem_col);
                int swizzled_idx = smem_row * BK + swizzled_col;
                uint32_t load_smem_b_ptr = smem_b_base_ptr + (swizzled_idx + smem_sel_next * SB_SIZE) * sizeof(T);
                CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);
            }
        }
        CP_ASYNC_COMMIT_GROUP();

        // Compute section - read with swizzle
        for (int TILE_K = 0; TILE_K < BK; TILE_K += WMMA_K)
        {
            // Read A matrix data - apply swizzle
            for (int i = 0; i < WARP_TILE_M; ++i)
            {
                int warp_smem_a_m = warp_m_id * WMMA_M * WARP_TILE_M + i * WMMA_M;
                int warp_smem_a_k = TILE_K;
                
                // Calculate original position
                int base_row = warp_smem_a_m + (lane_id % 16);
                int base_col = warp_smem_a_k + (lane_id / 16) * vec_size;
                
                // Apply swizzle to calculate actual storage position
                int swizzled_col = swizzle_permuted_A_j(base_row, base_col);
                T *lane_smem_a_ptr = smemA + base_row * BK + swizzled_col + smem_sel * SA_SIZE;
                
                uint32_t ptr = __cvta_generic_to_shared(lane_smem_a_ptr);
                LDMATRIX_X4(RA[i][0], RA[i][1], RA[i][2], RA[i][3], ptr);
            }
            
            // Read B matrix data - apply swizzle
            // 对于16x16x16 MMA，需要读取两组8列的数据
            for (int i = 0; i < WARP_TILE_N; ++i)
            {
                int warp_smem_b_n = warp_n_id * WMMA_N * WARP_TILE_N + i * WMMA_N;
                int warp_smem_b_k = TILE_K;
                
                // 第一组8列（0-7列）
                int base_row1 = warp_smem_b_n + (lane_id % 8);
                int base_col1 = warp_smem_b_k + (lane_id / 8) * vec_size;
                int swizzled_col1 = swizzle_permuted_B_j(base_row1, base_col1);
                T *lane_smem_b_ptr1 = smemB + base_row1 * BK + swizzled_col1 + smem_sel * SB_SIZE;
                uint32_t ptr1 = __cvta_generic_to_shared(lane_smem_b_ptr1);
                LDMATRIX_X2(RB[i][0], RB[i][1], ptr1);
                
                // 第二组8列（8-15列）
                int base_row2 = warp_smem_b_n + 8 + (lane_id % 8);
                int base_col2 = warp_smem_b_k + (lane_id / 8) * vec_size;
                int swizzled_col2 = swizzle_permuted_B_j(base_row2, base_col2);
                T *lane_smem_b_ptr2 = smemB + base_row2 * BK + swizzled_col2 + smem_sel * SB_SIZE;
                uint32_t ptr2 = __cvta_generic_to_shared(lane_smem_b_ptr2);
                LDMATRIX_X2(RB[i][2], RB[i][3], ptr2);
            }
            
            // Execute matrix multiplication with 16x16x16 MMA
            for (int i = 0; i < WARP_TILE_M; ++i)
            {
                for (int j = 0; j < WARP_TILE_N; ++j)
                {
                    MMA161616_BF16(RC[i][j][0], RC[i][j][1], RC[i][j][2], RC[i][j][3],
                                   RC[i][j][4], RC[i][j][5], RC[i][j][6], RC[i][j][7], 
                                   RA[i][0], RA[i][1], RA[i][2], RA[i][3], 
                                   RB[j][0], RB[j][1], RB[j][2], RB[j][3], 
                                   RC[i][j][0], RC[i][j][1], RC[i][j][2], RC[i][j][3],
                                   RC[i][j][4], RC[i][j][5], RC[i][j][6], RC[i][j][7]);
                }
            }
        }

        CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
        __syncthreads();
    }

    // Main loop end
    if ((K_STAGE - 2) > 0)
    {
        CP_ASYNC_WAIT_GROUP(0);
        __syncthreads();
    }

    // Compute remaining stages - apply swizzle
    for (int k_load = 0; k_load < K_STAGE - 1; ++k_load)
    {
        const int stage_sel = ((K / BK - (K_STAGE - 1) + k_load) % K_STAGE);
        for (int TILE_K = 0; TILE_K < BK; TILE_K += WMMA_K)
        {
            // Read A matrix data - apply swizzle
            for (int i = 0; i < WARP_TILE_M; ++i)
            {
                int warp_smem_a_m = warp_m_id * WMMA_M * WARP_TILE_M + i * WMMA_M;
                int warp_smem_a_k = TILE_K;
                
                // Calculate original position
                int base_row = warp_smem_a_m + (lane_id % 16);
                int base_col = warp_smem_a_k + (lane_id / 16) * vec_size;
                
                // Apply swizzle to calculate actual storage position
                int swizzled_col = swizzle_permuted_A_j(base_row, base_col);
                T *lane_smem_a_ptr = smemA + base_row * BK + swizzled_col + stage_sel * SA_SIZE;

                uint32_t ptr = __cvta_generic_to_shared(lane_smem_a_ptr);
                LDMATRIX_X4(RA[i][0], RA[i][1], RA[i][2], RA[i][3], ptr);
            }
            
            // Read B matrix data - apply swizzle
            // 对于16x16x16 MMA，需要读取两组8列的数据
            for (int i = 0; i < WARP_TILE_N; ++i)
            {
                int warp_smem_b_n = warp_n_id * WMMA_N * WARP_TILE_N + i * WMMA_N;
                int warp_smem_b_k = TILE_K;
                
                // 第一组8列（0-7列）
                int base_row1 = warp_smem_b_n + (lane_id % 8);
                int base_col1 = warp_smem_b_k + (lane_id / 8) * vec_size;
                int swizzled_col1 = swizzle_permuted_B_j(base_row1, base_col1);
                T *lane_smem_b_ptr1 = smemB + base_row1 * BK + swizzled_col1 + stage_sel * SB_SIZE;
                uint32_t ptr1 = __cvta_generic_to_shared(lane_smem_b_ptr1);
                LDMATRIX_X2(RB[i][0], RB[i][1], ptr1);
                
                // 第二组8列（8-15列）
                int base_row2 = warp_smem_b_n + 8 + (lane_id % 8);
                int base_col2 = warp_smem_b_k + (lane_id / 8) * vec_size;
                int swizzled_col2 = swizzle_permuted_B_j(base_row2, base_col2);
                T *lane_smem_b_ptr2 = smemB + base_row2 * BK + swizzled_col2 + stage_sel * SB_SIZE;
                uint32_t ptr2 = __cvta_generic_to_shared(lane_smem_b_ptr2);
                LDMATRIX_X2(RB[i][2], RB[i][3], ptr2);
            }
            
            // Execute matrix multiplication with 16x16x16 MMA
            for (int i = 0; i < WARP_TILE_M; ++i)
            {
                for (int j = 0; j < WARP_TILE_N; ++j)
                {
                    MMA161616_BF16(RC[i][j][0], RC[i][j][1], RC[i][j][2], RC[i][j][3],
                                   RC[i][j][4], RC[i][j][5], RC[i][j][6], RC[i][j][7], 
                                   RA[i][0], RA[i][1], RA[i][2], RA[i][3], 
                                   RB[j][0], RB[j][1], RB[j][2], RB[j][3], 
                                   RC[i][j][0], RC[i][j][1], RC[i][j][2], RC[i][j][3],
                                   RC[i][j][4], RC[i][j][5], RC[i][j][6], RC[i][j][7]);
                }
            }
        }
    }

#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i)
    {
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j)
        {
         
            const int tile_m0 = global_m_base + (warp_m_id * WMMA_M * WARP_TILE_M) + i * WMMA_M;
            const int tile_n0 = global_n_base + (warp_n_id * WMMA_N * WARP_TILE_N) + j * WMMA_N;

    
            int group = lane_id >> 2; // 0..7  → 行基
            int tid4 = lane_id & 3;   // 0..3  → 列基
            int row0 = group;         // 上半行
            int row1 = group + 8;     // 下半行
            
            // 对于16x16，列偏移需要覆盖16列，分为两组8列
            int col0 = 2 * tid4;      // 第一组8列中的左列
            int col1 = 2 * tid4 + 1;  // 第一组8列中的右列
            int col2 = 2 * tid4 + 8;  // 第二组8列中的左列
            int col3 = 2 * tid4 + 9;  // 第二组8列中的右列

            // 3. 取出8个累加结果（FP32）
            float v0 = reinterpret_cast<float*>(&RC[i][j][0])[0]; // 对应 (row0, col0)
            float v1 = reinterpret_cast<float*>(&RC[i][j][1])[0]; // 对应 (row0, col1)
            float v2 = reinterpret_cast<float*>(&RC[i][j][2])[0]; // 对应 (row1, col0)
            float v3 = reinterpret_cast<float*>(&RC[i][j][3])[0]; // 对应 (row1, col1)
            float v4 = reinterpret_cast<float*>(&RC[i][j][4])[0]; // 对应 (row0, col2)
            float v5 = reinterpret_cast<float*>(&RC[i][j][5])[0]; // 对应 (row0, col3)
            float v6 = reinterpret_cast<float*>(&RC[i][j][6])[0]; // 对应 (row1, col2)
            float v7 = reinterpret_cast<float*>(&RC[i][j][7])[0]; // 对应 (row1, col3)

            // 4. 折算成全局内存下标并越界保护
            if ((tile_m0 + row0) < M && (tile_n0 + col0) < N)
                C[(tile_m0 + row0) * N + (tile_n0 + col0)] = v0;
            if ((tile_m0 + row0) < M && (tile_n0 + col1) < N)
                C[(tile_m0 + row0) * N + (tile_n0 + col1)] = v1;
            if ((tile_m0 + row1) < M && (tile_n0 + col0) < N)
                C[(tile_m0 + row1) * N + (tile_n0 + col0)] = v2;
            if ((tile_m0 + row1) < M && (tile_n0 + col1) < N)
                C[(tile_m0 + row1) * N + (tile_n0 + col1)] = v3;
            if ((tile_m0 + row0) < M && (tile_n0 + col2) < N)
                C[(tile_m0 + row0) * N + (tile_n0 + col2)] = v4;
            if ((tile_m0 + row0) < M && (tile_n0 + col3) < N)
                C[(tile_m0 + row0) * N + (tile_n0 + col3)] = v5;
            if ((tile_m0 + row1) < M && (tile_n0 + col2) < N)
                C[(tile_m0 + row1) * N + (tile_n0 + col2)] = v6;
            if ((tile_m0 + row1) < M && (tile_n0 + col3) < N)
                C[(tile_m0 + row1) * N + (tile_n0 + col3)] = v7;
        }
    }
}