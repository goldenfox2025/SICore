#pragma once // 防止头文件被重复引用

#include <cublas_v2.h>
#include <cuda.h>
#include <cute/tensor.hpp> // CuTe核心库
#include <float.h>
#include <stdlib.h>

/**
 * @brief 使用CuTe实现的半精度通用矩阵乘法(HGEMM)核函数
 *
 * @tparam T 数据类型 (例如 half_t)
 * @tparam BM M维度上的线程块大小 (Block M)
 * @tparam BN N维度上的线程块大小 (Block N)
 * @tparam BK K维度上的线程块大小 (Block K)
 * @tparam kStage 软件流水线的阶段数(Stages), 用于隐藏数据加载延迟
 * @tparam TiledMMA CuTe的Tiled MMA对象, 定义了MMA操作的布局和形状
 * @tparam G2SCopyA 从全局内存(Global)到共享内存(Shared)的A矩阵拷贝操作
 * @tparam G2SCopyB 从全局内存到共享内存的B矩阵拷贝操作
 * @tparam SmemLayoutA A矩阵在共享内存中的布局
 * @tparam SmemLayoutB B矩阵在共享内存中的布局
 * @tparam SmemLayoutC C矩阵在共享内存中的布局 (用于Epilogue)
 * @tparam S2RCopyAtomA 从共享内存到寄存器(Register)的A矩阵拷贝原子操作
 * @tparam S2RCopyAtomB 从共享内存到寄存器的B矩阵拷贝原子操作
 * @tparam R2SCopyAtomC 从寄存器到共享内存的C矩阵拷贝原子操作 (用于Epilogue)
 * @tparam S2GCopyAtomC 从共享内存到全局内存的C矩阵拷贝原子操作 (用于Epilogue)
 * @tparam S2GCopyC 从共享内存到全局内存的C矩阵的Tiled Copy操作
 * @tparam BlockSwizzle 是否启用线程块Swizzle优化 (true/false)
 *
 * @param Aptr 输入矩阵A的全局内存指针
 * @param Bptr 输入矩阵B的全局内存指针
 * @param Dptr 输出矩阵D(结果矩阵)的全局内存指针
 * @param m 矩阵A, D的行数
 * @param n 矩阵B, D的列数
 * @param k 矩阵A的列数, 矩阵B的行数
 */
template <typename T, int BM, int BN, int BK, int kStage, typename TiledMMA,
          typename G2SCopyA, typename G2SCopyB, typename SmemLayoutA,
          typename SmemLayoutB, typename SmemLayoutC, typename S2RCopyAtomA,
          typename S2RCopyAtomB, typename R2SCopyAtomC, typename S2GCopyAtomC,
          typename S2GCopyC, const bool BlockSwizzle>
__global__ void
hgemm_mma_stages_block_swizzle_tn_cute_kernel(T *Aptr, T *Bptr, T *Dptr, int m,
                                              int n, int k) {
  using namespace cute;
  // 初始化共享内存
  extern __shared__ T shm_data[];

  T *Ashm = shm_data; // A矩阵的共享内存起始地址
  T *Bshm =
      shm_data + cute::cosize(SmemLayoutA{}); // B矩阵的共享内存起始地址

  // 初始化线程块索引
  int idx = threadIdx.x; // 线程在线程块内的唯一ID
  // BlockSwizzle为1时，启用块Swizzle, 重新计算块在N维度的逻辑索引
  int ix = ((int)BlockSwizzle) * blockIdx.z * gridDim.x + blockIdx.x;
  int iy = blockIdx.y;

  // 边界检查，如果当前块完全在矩阵范围外，则直接返回
  if (iy * BM >= m || ix * BN >= n)
    return;

  // 使用CuTe的Tensor来表示带有维度和步长的设备指针
  Tensor A = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k),
                         make_stride(k, Int<1>{})); // A(m, k), TN布局
  Tensor B = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k),
                         make_stride(k, Int<1>{})); // B(n, k), TN布局
  Tensor D = make_tensor(make_gmem_ptr(Dptr), make_shape(m, n),
                         make_stride(n, Int<1>{})); // D(m, n)

  // 将全局Tensor切片，得到当前线程块负责处理的小块(Tile)
  Tensor gA =
      local_tile(A, make_tile(Int<BM>{}, Int<BK>{}), make_coord(iy, _));
  Tensor gB =
      local_tile(B, make_tile(Int<BN>{}, Int<BK>{}), make_coord(ix, _));
  Tensor gD =
      local_tile(D, make_tile(Int<BM>{}, Int<BN>{}), make_coord(iy, ix));

  // 定义共享内存Tensor
  auto sA = make_tensor(make_smem_ptr(Ashm), SmemLayoutA{}); // (BM, BK, kStage)
  auto sB = make_tensor(make_smem_ptr(Bshm), SmemLayoutB{}); // (BN, BK, kStage)

  // 将TileA/TileB/TileC的MMA张量根据线程ID划分到每个线程的寄存器片段中
  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(threadIdx.x);
  auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0)); // A的寄存器片段
  auto tCrB = thr_mma.partition_fragment_B(gB(_, _, 0)); // B的寄存器片段
  auto tCrD = thr_mma.partition_fragment_C(gD);         // 累加器C(D)的寄存器片段
  clear(tCrD); // 清零累加器

  // ---- 数据拷贝：全局内存 -> 共享内存 ----
  G2SCopyA g2s_tiled_copy_a;
  auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(idx);
  // partition_S: 划分源张量(Source), 这里是全局内存gA
  auto tAgA_copy = g2s_thr_copy_a.partition_S(gA);
  // partition_D: 划分目标张量(Destination), 这里是共享内存sA
  auto tAsA_copy = g2s_thr_copy_a.partition_D(sA);

  G2SCopyB g2s_tiled_copy_b;
  auto g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(idx);
  auto tBgB_copy = g2s_thr_copy_b.partition_S(gB);
  auto tBsB_copy = g2s_thr_copy_b.partition_D(sB);

  // ---- 数据拷贝：共享内存 -> 寄存器 ----
  auto s2r_tiled_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
  auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(idx);
  auto tAsA = s2r_thr_copy_a.partition_S(sA); // 划分共享内存sA作为源
  auto tCrA_view =
      s2r_thr_copy_a.retile_D(tCrA); // 重新规划寄存器tCrA作为目标

  auto s2r_tiled_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
  auto s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(idx);
  auto tBsB = s2r_thr_copy_b.partition_S(sB);
  auto tCrB_view = s2r_thr_copy_b.retile_D(tCrB);

  /* ------ 主循环前的预取 (PREFETCH) ------ */
  int itile_to_read = 0; // K维度上，下一个要从全局内存读取的tile的索引
  int ismem_read = 0;    // 共享内存中，下一个要被读取的stage的索引
  int ismem_write = 0;   // 共享内存中，下一个要被写入的stage的索引

  // 预取kStage-1个tile的数据到共享内存，填满流水线
#pragma unroll
  for (int istage = 0; istage < kStage - 1; ++istage) {
    cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, istage),
               tAsA_copy(_, _, _, istage));
    cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, istage),
               tBsB_copy(_, _, _, istage));
    cp_async_fence(); // 确保之前的异步拷贝(cp.async)已提交

    ++itile_to_read;
    ++ismem_write;
  }

  // 等待，直到预取的数据至少有kStage-2个已经完成
  cp_async_wait<kStage - 2>();
  __syncthreads(); // 同步所有线程，确保共享内存数据对所有线程可见

  // 加载主循环的第一块数据：smem -> reg
  int ik = 0;
  cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik, ismem_read), tCrA_view(_, _, ik));
  cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik, ismem_read), tCrB_view(_, _, ik));

  // ------ K维度上的主循环 ------
  int ntile = k / BK; // K维度上的tile总数
#pragma unroll 1
  for (int itile = 0; itile < ntile; ++itile) {
    int nk = size<2>(tCrA); // MMA操作在K维度上的步数

#pragma unroll
    for (int ik = 0; ik < nk; ++ik) {
      int ik_next = (ik + 1) % nk;

      // 如果是K维度子块的最后一次迭代
      if (ik == nk - 1) {
        cp_async_wait<kStage - 2>(); // 等待下一个gmem->smem的数据块准备好
        __syncthreads();             // 同步，确保smem写入完成

        ismem_read =
            (ismem_read + 1) % kStage; // 更新下一轮要读取的smem stage索引
      }

      // 从共享内存加载下一次迭代需要的数据到寄存器
      cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik_next, ismem_read),
                 tCrA_view(_, _, ik_next));
      cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik_next, ismem_read),
                 tCrB_view(_, _, ik_next));

      // 如果是K维度子块的第一次迭代
      if (ik == 0) {
        // 从全局内存加载更后面的数据块到共享内存，维持流水线
        if (itile_to_read < ntile) {
          cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, itile_to_read),
                     tAsA_copy(_, _, _, ismem_write));
          cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, itile_to_read),
                     tBsB_copy(_, _, _, ismem_write));
          ++itile_to_read;
          ismem_write = (ismem_write + 1) % kStage; // 更新写入的smem stage索引
        }

        cp_async_fence(); // 提交异步拷贝
      }

      // 执行矩阵乘法累加操作 (D = A * B + D)
      cute::gemm(tiled_mma, tCrD, tCrA(_, _, ik), tCrB(_, _, ik), tCrD);
    } // for ik
  }

  /* ------ Epilogue: 写回结果 ------ */
  // 将寄存器中的结果通过共享内存写回到全局内存
  // 复用A矩阵的共享内存作为C矩阵的临时存储区(scratchpad)
  auto sC = make_tensor(sA(_, _, ismem_read).data(), SmemLayoutC{});

  // ---- 数据拷贝：寄存器 -> 共享内存 ----
  auto r2s_tiled_copy_c = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma);
  auto r2s_thr_copy_c = r2s_tiled_copy_c.get_slice(idx);
  auto tCrC_r2s = r2s_thr_copy_c.retile_S(tCrD);
  auto tCsC_r2s = r2s_thr_copy_c.partition_D(sC);

  // ---- 数据拷贝：共享内存 -> 全局内存 ----
  S2GCopyC s2g_tiled_copy_c;
  auto s2g_thr_copy_c = s2g_tiled_copy_c.get_thread_slice(idx);
  auto tCsC_s2g = s2g_thr_copy_c.partition_S(sC);
  auto tCgC_s2g = s2g_thr_copy_c.partition_D(gD);

  // 对拷贝张量的模式进行分组，简化拷贝循环
  auto tCgC_s2gx = group_modes<1, 3>(tCgC_s2g);
  auto tCrC_r2sx = group_modes<1, 3>(tCrC_r2s);

  int step = size<3>(tCsC_r2s); // 流水线步长
#pragma unroll
  for (int i = 0; i < size<1>(tCrC_r2sx); i += step) {
// 1. 寄存器 -> 共享内存
#pragma unroll
    for (int j = 0; j < step; ++j) {
      // 创建一个临时张量，以处理累加器(float)和输出(half)可能的数据类型差异
      auto t = make_tensor_like<T>(tCrC_r2sx(_, i + j));
      cute::copy(tCrC_r2sx(_, i + j), t); // 执行可能的类型转换

      cute::copy(r2s_tiled_copy_c, t, tCsC_r2s(_, 0, 0, j));
    }
    __syncthreads(); // 同步，确保数据写到共享内存

// 2. 共享内存 -> 全局内存
#pragma unroll
    for (int j = 0; j < step; ++j) {
      cute::copy(s2g_tiled_copy_c, tCsC_s2g(_, 0, 0, j), tCgC_s2gx(_, i + j));
    }
    __syncthreads(); // 同步，确保数据写到全局内存
  }
}

/**
 * @brief 启动HGEMM核函数的封装函数
 *
 * @tparam T 数据类型 (例如 half)
 * @tparam Stages 软件流水线阶段数，默认为2
 * @tparam BlockSwizzle 是否启用线程块Swizzle，默认为false
 * @param a 输入矩阵A的指针
 * @param b 输入矩阵B的指针
 * @param c 输出矩阵C的指针
 * @param M 矩阵M维度
 * @param N 矩阵N维度
 * @param K 矩阵K维度
 * @param swizzle_stride 线程块Swizzle的步长，仅在BlockSwizzle=true时有效
 */
template <typename T, const int Stages = 2, const bool BlockSwizzle = false>
void launch_hgemm_mma_stages_block_swizzle_tn_cute(T *a, T *b, T *c, int M,
                                                   int N, int K,
                                                   int swizzle_stride) {
  using namespace cute;

  // 定义线程块处理的tile大小
  auto BM = Int<128>{};
  auto BN = Int<256>{};
  auto BK = Int<32>{};
  auto KStage = Int<Stages>{}; // 流水线阶段数
  auto kSmemLayoutCBatch =
      Int<4>{}; // Epilogue中C矩阵在共享内存中的批处理大小

  // 定义A/B矩阵在共享内存中的布局 (SmemLayout)
  // Swizzle<3, 3, 3> 是一种数据混洗模式，通过改变数据在共享内存中的物理地址
  // 来减少访问冲突 (bank conflict)，提升带宽利用率。
  using SmemLayoutAtom = decltype(composition(
      Swizzle<3, 3, 3>{}, make_layout(make_shape(Int<8>{}, Int<BK>{}),
                                      make_stride(Int<BK>{}, Int<1>{}))));
  using SmemLayoutA = decltype(
      tile_to_shape(SmemLayoutAtom{},
                    make_shape(Int<BM>{}, Int<BK>{}, Int<KStage>{})));
  using SmemLayoutB = decltype(
      tile_to_shape(SmemLayoutAtom{},
                    make_shape(Int<BN>{}, Int<BK>{}, Int<KStage>{})));

  // 定义MMA (Matrix-Multiply-Accumulate) 操作
  // SM80_16x8x16_F16F16F16F16_TN 指定了使用安培架构(SM80)的Tensor Core
  // 其操作数形状为 (M,N,K) = (16,8,16)，数据类型为FP16
  using mma_op = SM80_16x8x16_F16F16F16F16_TN;
  using mma_traits = MMA_Traits<mma_op>;
  using mma_atom = MMA_Atom<mma_traits>;
  static constexpr int kMmaEURepeatM = 2; // MMA在M维度上重复2次
  static constexpr int kMmaEURepeatN = 2; // MMA在N维度上重复2次
  static constexpr int kMmaEURepeatK = 1; // MMA在K维度上不重复

  // 计算每个Warp处理的MMA tile的大小
  using mma_atom_shape = mma_traits::Shape_MNK;
  static constexpr int kMmaPM =
      1 * kMmaEURepeatM * get<0>(mma_atom_shape{}); // 1*2*16=32
  static constexpr int kMmaPN =
      2 * kMmaEURepeatN * get<1>(mma_atom_shape{}); // 2*2*8=32
  static constexpr int kMmaPK =
      1 * kMmaEURepeatK * get<2>(mma_atom_shape{}); // 1*1*16=16

  // 定义TiledMMA，它将多个MMA操作组合成一个更大的操作单元
  // MMA_EU_RepeatT 定义了线程组内的MMA重复布局 (一个warp-group/quad)
  // MMA_P_T 定义了线程块级别的MMA tile大小
  using MMA_EU_RepeatT = decltype(make_layout(
      make_shape(Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{},
                 Int<kMmaEURepeatK>{})));
  using MMA_P_T = Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>;
  using MMA = decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_P_T{}));

  // 定义从全局内存到共享内存的异步拷贝操作 (cp.async)
  using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
  using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
  using g2s_copy_atom = Copy_Atom<g2s_copy_traits, T>;
  // make_tiled_copy定义了拷贝操作的线程布局和值布局
  // Thr layout (32x4): 128个线程(一个线程块)分成32x4的网格进行拷贝
  // Val layout (1x8): 每个线程一次拷贝8个元素 (128 bits)
  using G2SCopyA = decltype(make_tiled_copy(
      g2s_copy_atom{},
      make_layout(make_shape(Int<32>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{})),
      make_layout(make_shape(Int<1>{}, Int<8>{}))));
  using G2SCopyB = G2SCopyA;

  // 定义从共享内存到寄存器的拷贝操作
  // 使用LDSM指令进行高效加载
  using s2r_copy_op = SM75_U32x4_LDSM_N;
  using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
  using s2r_copy_atom = Copy_Atom<s2r_copy_traits, T>;
  using S2RCopyAtomA = s2r_copy_atom;
  using S2RCopyAtomB = s2r_copy_atom;

  // 定义Epilogue(写回阶段)中使用到的布局和拷贝操作
  // C矩阵在共享内存中的布局
  using SmemLayoutAtomC = decltype(composition(
      Swizzle<3, 3, 3>{},
      make_layout(make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}),
                  make_stride(Int<kMmaPN>{}, Int<1>{}))));
  using SmemLayoutC = decltype(
      tile_to_shape(SmemLayoutAtomC{}, make_shape(Int<kMmaPM>{}, Int<kMmaPN>{},
                                                  Int<kSmemLayoutCBatch>{})));

  // 静态断言: 确保为C矩阵复用的共享内存空间足够大
  static_assert(
      size<0>(SmemLayoutA{}) * size<1>(SmemLayoutA{}) >= size(SmemLayoutC{}),
      "C shared memory request is larger than A's one pipe");

  // 定义寄存器->共享内存，共享内存->全局内存的拷贝原子
  using R2SCopyAtomC = Copy_Atom<UniversalCopy<int>, T>;
  using S2GCopyAtomC = Copy_Atom<UniversalCopy<cute::uint128_t>, T>;
  // 定义共享内存->全局内存的Tiled Copy
  using S2GCopyC = decltype(make_tiled_copy(
      S2GCopyAtomC{},
      make_layout(make_shape(Int<32>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{})),
      make_layout(make_shape(Int<1>{}, Int<8>{}))));

  // 计算Grid维度
  int BX = (N + BN - 1) / BN;
  int BY = (M + BM - 1) / BM;
  // 如果启用BlockSwizzle，则增加Z维度用于Swizzle
  int BZ = BlockSwizzle ? (N + (swizzle_stride)-1) / (swizzle_stride) : 1;
  BX = BlockSwizzle ? (BX + BZ - 1) / BZ : BX;

  dim3 block(size(MMA{})); // 线程块大小，通常是128或256
  dim3 grid(BX, BY, BZ);   // 网格大小

  // 计算所需的共享内存大小
  // A和B的共享内存是同时存在的，C的共享内存是复用A/B的，所以取两者最大值
  static constexpr int shm_size_AB =
      cute::cosize(SmemLayoutA{}) + cute::cosize(SmemLayoutB{});
  static constexpr int shm_size_C = cute::cosize(SmemLayoutC{});
  static constexpr int kShmSize =
      cute::max(shm_size_AB, shm_size_C) * sizeof(T);

  int shm_size = kShmSize;

  // 设置核函数的动态共享内存大小属性
  cudaFuncSetAttribute(
      hgemm_mma_stages_block_swizzle_tn_cute_kernel<
          T, BM, BN, BK, KStage, MMA, G2SCopyA, G2SCopyB, SmemLayoutA,
          SmemLayoutB, SmemLayoutC, S2RCopyAtomA, S2RCopyAtomB, R2SCopyAtomC,
          S2GCopyAtomC, S2GCopyC, BlockSwizzle>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

  // 启动核函数
  hgemm_mma_stages_block_swizzle_tn_cute_kernel<
      T, BM, BN, BK, KStage, MMA, G2SCopyA, G2SCopyB, SmemLayoutA, SmemLayoutB,
      SmemLayoutC, S2RCopyAtomA, S2RCopyAtomB, R2SCopyAtomC, S2GCopyAtomC,
      S2GCopyC, BlockSwizzle><<<grid, block, shm_size>>>(a, b, c, M, N, K);
}