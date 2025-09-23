#pragma once

#include <cuda_runtime.h>
#include <cute/config.hpp>
#include <cute/layout.hpp>
#include <cute/pointer.hpp>
#include <cute/stride.hpp>
#include <cute/tensor.hpp>
#include <cute/tensor_impl.hpp>

using namespace cute;

#define UNIT_BLK_SIZE 16


template <typename T , int BLK_M , int BLK_N , typename TiledCopyA , typename TiledCopyB , typename SmemLayoutA , typename SmemLayoutB>
__global__
void mat_trans_cute_vectorized_kernel(
    const T* matrixA,
    T* matrixO,
    int M , int N,
    TiledCopyA copy_a, TiledCopyB copy_b,
    SmemLayoutA sA_layout, SmemLayoutB sB_layout
){
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    auto ma = make_tensor(make_gmem_ptr(matrixA) ,make_layout(make_shape(M, N), GenRowMajor{}));
    auto mb = make_tensor(make_gmem_ptr(matrixO) ,make_layout(make_shape(N, M), GenRowMajor{}));

    auto gA = local_tile(ma, make_shape(Int<BLK_M>{}, Int<BLK_N>{}),make_coord(bx, by));
    auto gB = local_tile(mb, make_shape(Int<BLK_N>{}, Int<BLK_M>{}), make_coord(by, bx));

    __shared__ T smem[BLK_M * BLK_N];

    auto sA = make_tensor(make_smem_ptr(smem), sA_layout);
    auto sB = make_tensor(make_smem_ptr(smem), sB_layout);

    auto thr_copy_a = copy_a.get_slice(tx);
    Tensor tAgA = thr_copy_a.partition_S(gA);
    Tensor tAsA = thr_copy_a.partition_D(sA);

    auto thr_copy_b = copy_b.get_slice(tx);
    Tensor tBsB = thr_copy_b.partition_S(sB);
    Tensor tBgB = thr_copy_b.partition_D(gB);
    copy(copy_a, tAgA, tAsA);
    __syncthreads();
    copy(copy_b, tBsB, tBgB);

}

void mat_transpose_cute_row_cvectorized(float* x, float* y ,int M , int N) {
  const int BM = UNIT_BLK_SIZE * 4;
  const int BN = UNIT_BLK_SIZE;

  assert(M % 4 == 0);
  assert(N % 4 == 0);

  static_assert(BM % 4 == 0);
  static_assert(BN % 4 == 0);
  

  auto tile_copy_a = make_tiled_copy(
      Copy_Atom<AutoVectorizingCopy, float>{},
      make_layout(make_shape(Int<BM / 4>{}, Int<BN>{}), GenRowMajor{}),
      make_layout(make_shape(Int<4>{}, Int<1>{}), GenRowMajor{}));

  auto tile_copy_b = make_tiled_copy(
      Copy_Atom<AutoVectorizingCopy, float>{},
      make_layout(make_shape(Int<BN>{}, Int<BM / 4>{}), GenRowMajor{}),
      make_layout(make_shape(Int<1>{}, Int<4>{}), GenRowMajor{}));

  auto sA_layout = make_layout(make_shape(Int<BM>{}, Int<BN>{}), GenRowMajor{});
  auto sB_layout = make_layout(make_shape(Int<BN>{}, Int<BM>{}), GenColMajor{});

  static_assert(size(tile_copy_a) == size(tile_copy_b));
  dim3 block(size(tile_copy_a));
  dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);

  mat_trans_cute_vectorized_kernel<float, BM, BN, decltype(tile_copy_a),
                                   decltype(tile_copy_b), decltype(sA_layout),
                                   decltype(sB_layout)><<<grid, block>>>(
      x, y, M, N, tile_copy_a, tile_copy_b, sA_layout, sB_layout);
}