#pragma once

#include <cuda_runtime.h>
#include <cute/config.hpp>
#include <cute/layout.hpp>
#include <cute/pointer.hpp>
#include <cute/pointer_flagged.hpp>
#include <cute/stride.hpp>
#include <cute/tensor.hpp>
#include <cute/tensor_impl.hpp>


using namespace cute;

#define UNIT_BLK_SIZE 16

template <typename T , int BLK_M , int BLK_N , typename ThreadLayoutA, typename ThreadLayoutB>
__global__ void mat_trans_cute_reg_kernel(
    const T* pA , T* pB , int M , int N , ThreadLayoutA tA , ThreadLayoutB tB
){

    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    auto ma = make_tensor(make_gmem_ptr(pA) , make_layout(make_shape(M,N), GenRowMajor{}));
    auto mb = make_tensor(make_gmem_ptr(pB) , make_layout(make_shape(N , M) , GenRowMajor{}));


    auto ga = local_tile(ma , make_shape(Int<BLK_M>{} , Int<BLK_N>{}) , make_coord(bx , by));
    auto gb = local_tile(mb , make_shape(Int<BLK_N>{} , Int<BLK_M>{}) , make_coord(by , bx));

    auto ca = local_tile(make_identity_tensor(ma.shape()) , make_shape(Int<BLK_M>{} , Int<BLK_N>{}) , make_coord(bx , by));


    Tensor tAga = local_partition(ga , tA , tx);
    Tensor tBgb = local_partition(gb , tB , tx); 

    Tensor tAcA = local_partition(ca , tA , tx);

    //for bound checks -> represents a block
    Tensor tApA = make_tensor<bool>(tAcA.shape() , tAcA.stride());

    CUTE_UNROLL
    for(int i = 0 ; i < size<0>(tApA) ; i++){
        CUTE_UNROLL
        for(int j = 0 ; j < size<1>(tApA) ; j++){
            tApA(i , j) = get<0>(tAcA(i,j)) < M && get<1>(tAcA(i,j)) < N;
        }
    }

    copy_if(tApA , tAga , tBgb);
}

// void mat_transpose_cute_row2col_reg(torch::Tensor x, torch::Tensor y) {
void mat_transpose_cute_row2col_reg(float*  x, float* y , int M , int N) {

  const int BM = UNIT_BLK_SIZE;
  const int BN = UNIT_BLK_SIZE;
  
  
  auto tA = make_layout(make_shape(Int<BM>{}, Int<BN>{}), GenColMajor{});
  auto tB = make_layout(make_shape(Int<BN>{}, Int<BM>{}), GenRowMajor{});
  
  static_assert(size(tA) == size(tB));
  
  dim3 block(size(tA));
  dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);

  mat_trans_cute_reg_kernel<float, BM, BN, decltype(tA), decltype(tB)>
      <<<grid, block>>>(x, y, M, N, tA, tB);
    //   <<<grid, block>>>(x.data_ptr<float>(), y.data_ptr<float>(), M, N, tA, tB);
}

