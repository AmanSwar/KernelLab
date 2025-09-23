#pragma once

#include <cuda_runtime.h>
#include <cute/config.hpp>
#include <cute/stride.hpp>
#include <cute/tensor.hpp>
#include <cute/layout.hpp>
#include <cute/pointer.hpp>
#include <cute/tensor_impl.hpp>


using namespace cute;

#define UNIT_BLK_SIZE 16


template <typename T , int BLK_M , int BLK_N , typename ThreadLayoutA , typename ThreadLayoutB , typename SmemLayoutA , typename SmemLayoutB>
__global__ void
mat_trans_cute_smem_kernel(
    const T *matrixA,
    T *matrixO,
    int M , int N,
    ThreadLayoutA tA,
    ThreadLayoutB tB,
    SmemLayoutA sa_layout,
    SmemLayoutB sb_layout
){
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int by =blockIdx.y;

    auto tensorA = make_tensor(make_gmem_ptr(matrixA) , make_layout(make_shape(M , N) , GenRowMajor{}));
    auto tensorO = make_tensor(make_gmem_ptr(matrixO) , make_layout(make_shape(N , M) , GenRowMajor{}));

    auto tileA = local_tile(tensorA , make_shape(Int<BLK_M>{}, Int<BLK_N>{}) , make_coord(bx , by));
    auto tileO = local_tile(tensorO , make_shape(Int<BLK_N>{}, Int<BLK_M>{}) , make_coord(by , bx));

    auto cA = local_tile(make_identity_tensor(tensorA.shape()) , make_shape(Int<BLK_M>{} , Int<BLK_N>{}) , make_coord(bx , by));
    auto cO = local_tile(make_identity_tensor(tensorO.shape()) , make_shape(Int<BLK_N>{} , Int<BLK_M>{}) , make_coord(by , bx));

    __shared__ T smem[BLK_M * BLK_N];

    auto smem_tensorA = make_tensor(make_smem_ptr(smem) , sa_layout);
    auto smem_tensorO = make_tensor(make_smem_ptr(smem) , sb_layout);

    //global
    Tensor tAgA = local_partition(tileA , tA , tx);
    Tensor tBgB = local_partition(tileO , tB  ,tx);

    //smem
    Tensor tAsA = local_partition(smem_tensorA , tA , tx);
    Tensor tOsO = local_partition(smem_tensorO , tB , tx);

    Tensor tAcA = local_partition(cA , tA , tx);
    Tensor tOcO = local_partition(cO , tB , tx);

    Tensor tApA = make_tensor<bool>(tAcA.shape() , tAcA.stride());
    Tensor tOpO = make_tensor<bool>(tOcO.shape() , tOcO.stride());

    CUTE_UNROLL
    for(int i = 0; i < size<0>(tApA) ;i++){
        CUTE_UNROLL
        for(int j = 0 ; j < size<1>(tApA); j++){
            tApA(i , j) = get<0>(tAcA(i,j)) < M && get<1>(tAcA(i , j)) < N;
        }
    }

    CUTE_UNROLL
    for(int i = 0; i < size<0>(tOpO) ; i++){
        CUTE_UNROLL
        for (int j = 0; j < size<1>(tOpO); j++) {
          tOpO(i, j) = get<0>(tOcO(i, j)) < N && get<1>(tOcO(i, j)) < M;
        }
    }

    copy_if(tApA , tAgA , tAsA);
    __syncthreads();
    copy_if(tOpO , tOsO , tBgB);
}

void mat_transpose_cute_row_smem(float* x , float* y, int M , int N) {

  const int BM = UNIT_BLK_SIZE;
  const int BN = UNIT_BLK_SIZE;
  
  auto tA = make_layout(make_shape(Int<BM>{}, Int<BN>{}), GenRowMajor{});
  auto tB = make_layout(make_shape(Int<BN>{}, Int<BM>{}), GenRowMajor{});

  auto sA_layout = make_layout(make_shape(Int<BM>{}, Int<BN>{}), GenRowMajor{});
  auto sB_layout = make_layout(make_shape(Int<BN>{}, Int<BM>{}), GenColMajor{});

  static_assert(size(tA) == size(tB));

  dim3 block(size(tA));
  dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
  mat_trans_cute_smem_kernel<float, BM, BN, decltype(tA), decltype(tB),
                             decltype(sA_layout), decltype(sB_layout)>
      <<<grid, block>>>(x, y, M, N, tA, tB, sA_layout, sB_layout);
}