#pragma once

#include <cublas_v2.h>
#include <cuda.h>
#include <cute/tensor.hpp>
#include <float.h>
#include <stdlib.h>
#include <type_traits>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

/**
 * @brief 超级简单的CUTE GEMM kernel，用于验证基本概念
 * 直接计算 C[i][j] = sum_k A[i][k] * B[j][k]，与naive kernel完全一致
 */
template <typename T>
__global__ void simple_cute_kernel(T *Aptr, T *Bptr, T *Cptr, int M, int N, int K) {
  using namespace cute;
  
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (i >= M || j >= N) return;
  
  // 创建最简单的tensors，直接按照内存存储方式
  // A是M×K，存储为A[row*K + col]
  // B是N×K，存储为B[row*K + col] (但在计算中，B[j][k]表示B的第j行第k列)
  // C是M×N，存储为C[row*N + col]
  
  // 使用最基本的stride定义
  Tensor A = make_tensor(make_gmem_ptr(Aptr), make_shape(M, K), make_stride(K, 1));
  Tensor B = make_tensor(make_gmem_ptr(Bptr), make_shape(N, K), make_stride(K, 1));
  Tensor C = make_tensor(make_gmem_ptr(Cptr), make_shape(M, N), make_stride(N, 1));
  
  // 直接计算，与naive kernel一模一样
  float sum = 0.0f;  // 使用float精度计算，与naive kernel一致
  for (int k = 0; k < K; ++k) {
    sum += static_cast<float>(A(i, k)) * static_cast<float>(B(j, k));
  }
  C(i, j) = static_cast<T>(sum);
}

/**
 * @brief 启动简单CUTE kernel
 */
template <typename T>
void launch_simple_cute(T *a, T *b, T *c, int M, int N, int K) {
  dim3 block(16, 16);
  dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
  
  simple_cute_kernel<T><<<grid, block>>>(a, b, c, M, N, K);
  cudaDeviceSynchronize();
}