#pragma once

#include <cublas_v2.h>
#include <cuda.h>
#include <cute/tensor.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/atom/mma_atom.hpp>
#include <float.h>
#include <stdlib.h>
#include <type_traits>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// Helper function for address conversion
__device__ __forceinline__ uint32_t smem_u32addr(const void* smem_ptr) {
    uint32_t addr;
    asm volatile("{.reg .u64 u64addr; cvta.to.shared.u64 u64addr, %1; cvt.u32.u64 %0, u64addr;}\n"
                 : "=r"(addr) : "l"(smem_ptr));
    return addr;
}

#define __cvta_generic_to_shared(ptr) smem_u32addr(ptr)


template <typename T, int BM, int BN, int BK, int kStage, typename TiledMMA,
          typename G2SCopyA, typename G2SCopyB, typename SmemLayoutA,
          typename SmemLayoutB, typename SmemLayoutC, typename S2RCopyAtomA,
          typename S2RCopyAtomB, typename R2SCopyAtomC, typename S2GCopyAtomC,
          typename S2GCopyC, const bool BlockSwizzle>
__global__ void
gemm_mma_stages_block_swizzle_tn_cute2_kernel(T *Aptr, T *Bptr, T *Dptr, int m,
                                               int n, int k) {
  using namespace cute;
  // 初始化共享内存
  extern __shared__ T shm_data[];

  T *Ashm = shm_data;
  T *Bshm = shm_data + cute::cosize(SmemLayoutA{});

  // 初始化线程块索引
  int idx = threadIdx.x;
  int ix = ((int)BlockSwizzle) * blockIdx.z * gridDim.x + blockIdx.x;
  int iy = blockIdx.y;

  // 边界检查
  if (iy * BM >= m || ix * BN >= n)
    return;

  // 关键修正：使用正确的矩阵布局定义
  // 这里的关键是匹配我们验证过的简单版本的tensor定义
  Tensor A = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k),
                         make_stride(k, Int<1>{})); // A(m,k) row-major
  Tensor B = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k),
                         make_stride(k, Int<1>{})); // B(n,k) row-major
  Tensor D = make_tensor(make_gmem_ptr(Dptr), make_shape(m, n),
                         make_stride(n, Int<1>{})); // D(m,n) row-major

  // 将全局Tensor切片
  Tensor gA = local_tile(A, make_tile(Int<BM>{}, Int<BK>{}), make_coord(iy, _));
  Tensor gB = local_tile(B, make_tile(Int<BN>{}, Int<BK>{}), make_coord(ix, _));
  Tensor gD = local_tile(D, make_tile(Int<BM>{}, Int<BN>{}), make_coord(iy, ix));

  // 定义共享内存Tensor
  auto sA = make_tensor(make_smem_ptr(Ashm), SmemLayoutA{});
  auto sB = make_tensor(make_smem_ptr(Bshm), SmemLayoutB{});

  // MMA分割
  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(threadIdx.x);
  auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0));
  auto tCrB = thr_mma.partition_fragment_B(gB(_, _, 0));
  auto tCrD = thr_mma.partition_fragment_C(gD);
  clear(tCrD);

  // 数据拷贝设置
  G2SCopyA g2s_tiled_copy_a;
  auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(idx);
  auto tAgA_copy = g2s_thr_copy_a.partition_S(gA);
  auto tAsA_copy = g2s_thr_copy_a.partition_D(sA);

  G2SCopyB g2s_tiled_copy_b;
  auto g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(idx);
  auto tBgB_copy = g2s_thr_copy_b.partition_S(gB);
  auto tBsB_copy = g2s_thr_copy_b.partition_D(sB);

  auto s2r_tiled_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
  auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(idx);
  auto tAsA = s2r_thr_copy_a.partition_S(sA);
  auto tCrA_view = s2r_thr_copy_a.retile_D(tCrA);

  auto s2r_tiled_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
  auto s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(idx);
  auto tBsB = s2r_thr_copy_b.partition_S(sB);
  auto tCrB_view = s2r_thr_copy_b.retile_D(tCrB);

  // 主循环前的预取
  int itile_to_read = 0;
  int ismem_read = 0;
  int ismem_write = 0;

#pragma unroll
  for (int istage = 0; istage < kStage - 1; ++istage) {
    cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, istage),
               tAsA_copy(_, _, _, istage));
    cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, istage),
               tBsB_copy(_, _, _, istage));
    cp_async_fence();

    ++itile_to_read;
    ++ismem_write;
  }

  cp_async_wait<kStage - 2>();
  __syncthreads();

  // 加载第一块数据
  int ik = 0;
  cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik, ismem_read), tCrA_view(_, _, ik));
  cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik, ismem_read), tCrB_view(_, _, ik));

  // 主循环
  int ntile = k / BK;
#pragma unroll 1
  for (int itile = 0; itile < ntile; ++itile) {
    int nk = size<2>(tCrA);

#pragma unroll
    for (int ik = 0; ik < nk; ++ik) {
      int ik_next = (ik + 1) % nk;

      if (ik == nk - 1) {
        cp_async_wait<kStage - 2>();
        __syncthreads();
        ismem_read = (ismem_read + 1) % kStage;
      }

      cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik_next, ismem_read),
                 tCrA_view(_, _, ik_next));
      cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik_next, ismem_read),
                 tCrB_view(_, _, ik_next));

      if (ik == 0) {
        if (itile_to_read < ntile) {
          cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, itile_to_read),
                     tAsA_copy(_, _, _, ismem_write));
          cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, itile_to_read),
                     tBsB_copy(_, _, _, ismem_write));
          ++itile_to_read;
          ismem_write = (ismem_write + 1) % kStage;
        }
        cp_async_fence();
      }

      // 执行MMA操作
      cute::gemm(tiled_mma, tCrD, tCrA(_, _, ik), tCrB(_, _, ik), tCrD);
    }
  }


  auto sC = make_tensor(sA(_, _, ismem_read).data(), SmemLayoutC{});

  auto r2s_tiled_copy_c = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma);
  auto r2s_thr_copy_c = r2s_tiled_copy_c.get_slice(idx);
  auto tCrC_r2s = r2s_thr_copy_c.retile_S(tCrD);
  auto tCsC_r2s = r2s_thr_copy_c.partition_D(sC);

  S2GCopyC s2g_tiled_copy_c;
  auto s2g_thr_copy_c = s2g_tiled_copy_c.get_thread_slice(idx);
  auto tCsC_s2g = s2g_thr_copy_c.partition_S(sC);
  auto tCgC_s2g = s2g_thr_copy_c.partition_D(gD);

  auto tCgC_s2gx = group_modes<1, 3>(tCgC_s2g);
  auto tCrC_r2sx = group_modes<1, 3>(tCrC_r2s);

  int step = size<3>(tCsC_r2s);
#pragma unroll
  for (int i = 0; i < size<1>(tCrC_r2sx); i += step) {
#pragma unroll
    for (int j = 0; j < step; ++j) {
      // 改进的类型转换处理
      if constexpr (std::is_same_v<T, nv_bfloat16> || std::is_same_v<T, half>) {
        // 对于半精度类型，创建临时tensor进行适当的类型转换
        auto t = make_tensor_like<T>(tCrC_r2sx(_, i + j));
        
        // 手动进行element-wise拷贝和类型转换
        CUTE_UNROLL
        for (int elem = 0; elem < size(t); ++elem) {
          if constexpr (std::is_same_v<T, nv_bfloat16>) {
            t(elem) = __float2bfloat16(__bfloat162float(tCrC_r2sx(_, i + j)(elem)));
          } else {
            t(elem) = __float2half(__half2float(tCrC_r2sx(_, i + j)(elem)));
          }
        }
        
        cute::copy(r2s_tiled_copy_c, t, tCsC_r2s(_, 0, 0, j));
      } else {
        // 对于其他类型，直接拷贝
        cute::copy(r2s_tiled_copy_c, tCrC_r2sx(_, i + j), tCsC_r2s(_, 0, 0, j));
      }
    }
    __syncthreads();

#pragma unroll
    for (int j = 0; j < step; ++j) {
      cute::copy(s2g_tiled_copy_c, tCsC_s2g(_, 0, 0, j), tCgC_s2gx(_, i + j));
    }
    __syncthreads();
  }
}

/**
 * @brief 启动修正版高速CUTE GEMM核函数
 */
template <typename T, const int Stages = 2, const bool BlockSwizzle = false>
void launch_gemm_mma_stages_block_swizzle_tn_cute2(T *a, T *b, T *c, int M,
                                                    int N, int K,
                                                    int swizzle_stride) {
  using namespace cute;

  // 使用与原始高速版本相同的配置
  auto BM = Int<128>{};
  auto BN = Int<128>{};
  auto BK = Int<32>{};
  auto KStage = Int<Stages>{};
  auto kSmemLayoutCBatch = Int<4>{};

  // 定义布局 - 与原始版本相同
  using SmemLayoutAtom = decltype(composition(
      Swizzle<3, 3, 3>{}, make_layout(make_shape(Int<8>{}, Int<BK>{}),
                                      make_stride(Int<BK>{}, Int<1>{}))));
  using SmemLayoutA = decltype(
      tile_to_shape(SmemLayoutAtom{},
                    make_shape(Int<BM>{}, Int<BK>{}, Int<KStage>{})));
  using SmemLayoutB = decltype(
      tile_to_shape(SmemLayoutAtom{},
                    make_shape(Int<BN>{}, Int<BK>{}, Int<KStage>{})));

  // 使用专门的bfloat16混合精度MMA：输入bfloat16，累加float精度
  using mma_op = SM80_16x8x16_F32BF16BF16F32_TN; // bfloat16输入，float32累加
  using mma_traits = MMA_Traits<mma_op>;
  using mma_atom = MMA_Atom<mma_traits>;
  static constexpr int kMmaEURepeatM = 2;
  static constexpr int kMmaEURepeatN = 2;
  static constexpr int kMmaEURepeatK = 1;

  using mma_atom_shape = mma_traits::Shape_MNK;
  static constexpr int kMmaPM = 1 * kMmaEURepeatM * get<0>(mma_atom_shape{});
  static constexpr int kMmaPN = 2 * kMmaEURepeatN * get<1>(mma_atom_shape{});
  static constexpr int kMmaPK = 1 * kMmaEURepeatK * get<2>(mma_atom_shape{});

  using MMA_EU_RepeatT = decltype(make_layout(
      make_shape(Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{},
                 Int<kMmaEURepeatK>{})));
  using MMA_P_T = Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>;
  using MMA = decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_P_T{}));

  // Copy操作定义 - 与原始版本相同
  using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
  using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
  using g2s_copy_atom = Copy_Atom<g2s_copy_traits, T>;
  using G2SCopyA = decltype(make_tiled_copy(
      g2s_copy_atom{},
      make_layout(make_shape(Int<32>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{})),
      make_layout(make_shape(Int<1>{}, Int<8>{}))));
  using G2SCopyB = G2SCopyA;

  using s2r_copy_op = SM75_U32x4_LDSM_N;
  using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
  using s2r_copy_atom = Copy_Atom<s2r_copy_traits, T>;
  using S2RCopyAtomA = s2r_copy_atom;
  using S2RCopyAtomB = s2r_copy_atom;

  using SmemLayoutAtomC = decltype(composition(
      Swizzle<3, 3, 3>{},
      make_layout(make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}),
                  make_stride(Int<kMmaPN>{}, Int<1>{}))));
  using SmemLayoutC = decltype(
      tile_to_shape(SmemLayoutAtomC{}, make_shape(Int<kMmaPM>{}, Int<kMmaPN>{},
                                                  Int<kSmemLayoutCBatch>{})));

  static_assert(
      size<0>(SmemLayoutA{}) * size<1>(SmemLayoutA{}) >= size(SmemLayoutC{}),
      "C shared memory request is larger than A's one pipe");

  using R2SCopyAtomC = Copy_Atom<UniversalCopy<int>, T>;
  using S2GCopyAtomC = Copy_Atom<UniversalCopy<cute::uint128_t>, T>;
  using S2GCopyC = decltype(make_tiled_copy(
      S2GCopyAtomC{},
      make_layout(make_shape(Int<32>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{})),
      make_layout(make_shape(Int<1>{}, Int<8>{}))));

  // Grid计算
  int BX = (N + BN - 1) / BN;
  int BY = (M + BM - 1) / BM;
  int BZ = BlockSwizzle ? (N + (swizzle_stride)-1) / (swizzle_stride) : 1;
  BX = BlockSwizzle ? (BX + BZ - 1) / BZ : BX;

  dim3 block(size(MMA{}));
  dim3 grid(BX, BY, BZ);

  static constexpr int shm_size_AB =
      cute::cosize(SmemLayoutA{}) + cute::cosize(SmemLayoutB{});
  static constexpr int shm_size_C = cute::cosize(SmemLayoutC{});
  static constexpr int kShmSize =
      cute::max(shm_size_AB, shm_size_C) * sizeof(T);

  int shm_size = kShmSize;

  cudaFuncSetAttribute(
      gemm_mma_stages_block_swizzle_tn_cute2_kernel<
          T, BM, BN, BK, KStage, MMA, G2SCopyA, G2SCopyB, SmemLayoutA,
          SmemLayoutB, SmemLayoutC, S2RCopyAtomA, S2RCopyAtomB, R2SCopyAtomC,
          S2GCopyAtomC, S2GCopyC, BlockSwizzle>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

  gemm_mma_stages_block_swizzle_tn_cute2_kernel<
      T, BM, BN, BK, KStage, MMA, G2SCopyA, G2SCopyB, SmemLayoutA, SmemLayoutB,
      SmemLayoutC, S2RCopyAtomA, S2RCopyAtomB, R2SCopyAtomC, S2GCopyAtomC,
      S2GCopyC, BlockSwizzle><<<grid, block, shm_size>>>(a, b, c, M, N, K);
}