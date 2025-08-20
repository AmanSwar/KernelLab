#include <__clang_cuda_builtin_vars.h>
#include <cassert>
#include <cuda_runtime.h>
#include <cute/layout.hpp>
#include <cute/pointer.hpp>
#include <cute/pointer_flagged.hpp>
#include <cute/tensor.hpp>
#include <cute/tensor_impl.hpp>

#include "../util.cuh"

// softmax kernel
__device__ void softmax_kernel(float *matrix, int M, int N) {
  int global_index = blockIdx.x * blockDim.x + threadIdx.x;
  int local_index = threadIdx.x;

  int row_start = global_index * N; // 0 * N , 1 * N;

  for (int i = 0; i < N; i += blockDim.x)
}

// attention kernel
template <class Query, class Key, class Score, class Value,
          class QuerySmemLayout, class KeySmemLayout, class ValueSmemLayout,
          class QueryThreadLayout, class KeyThreadLayout,
          class ScoreThreadLayout>
__global__ void
self_attention(Query const *Q, Key const *K, Score const *SCORE, Value const *V,
               int const seq_length, int const ndim, QuerySmemLayout sq_layout,
               KeySmemLayout sk_layout, ValueSmemLayout sv_layout) {

  using namespace cute;
  // Tensor for global mem
  Tensor global_query =
      make_tensor(make_gmem_ptr(Q), make_shape(seq_length, ndim), ndim);
  Tensor global_key =
      make_tensor(make_gmem_ptr(K), make_shape(seq_length, ndim), seq_length);
  Tensor global_score =
      make_tensor(make_gmem_ptr(SCORE), make_shape(seq_length, seq_length));
  Tensor global_value =
      make_tensor(make_gmem_ptr(V), make_shape(seq_length, ndim), ndim);

  // block size
  auto block_seqLength = Int<128>{};
  auto block_ndim = Int<8>{};

  // tiler for extracting blocks
  auto tile_shape = make_shape(block_seqLength, block_seqLength, block_ndim);
  auto tile_coord = make_coord(blockIdx.x, blockIdx.y, _);

  // divide into tiles
  Tensor query_tile =
      make_tile(global_query, tile_shape, tile_coord, Step<_1, X, _1>{});
  Tensor key_tile = make_tile(global_key, tile_shape, tile_coord, Shape<X, _1, _1>{});
  Tensor score_tile =
      make_tile(global_score, tile_shape, tile_coord, Shape<_1, _1, X>{});

  // shared mem
  __shared__ Value smem_query[cosize_v<QuerySmemLayout>];
  __shared__ Value smem_key[cosize_v<KeySmemLayout>];

  Tensor query_smem = make_tensor(make_smem_ptr(smem_query), sq_layout);
  Tensor key_smem = make_tensor(make_smem_ptr(smem_key), sk_layout);

  Tensor threadQ_globalQ = local_partition()
}