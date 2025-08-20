// device_cutlass_gemm_f16.cu
// Compile (example):
// nvcc -std=c++14 -I/path/to/cutlass/include -arch=sm_80
// device_cutlass_gemm_f16.cu -o device_cutlass_gemm_f16

#include <cstdio>
#include <cuda_fp16.h>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/epilogue.h"
#include "cutlass/gemm/block_task.h" // threadblock-level block_task (older / stable pattern)
#include "cutlass/gemm/gemm.h"
#include "cutlass/half.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/matrix_ref.h"

using ElementA = cutlass::half_t; // FP16 inputs
using ElementB = cutlass::half_t;
using ElementC = cutlass::half_t; // output as FP16 (could be float)
using ElementAccum = float;       // accumulation in FP32

// --- Block / Warp tile choices (example) ---
// Tune these according to your problem / GPU. These match common CUTLASS tile
// sizes.
constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 128;
constexpr int BLOCK_K = 32;

// Thread-level subdivision (these parameters correspond to CUTLASS block_task
// policy template args)
constexpr int THREADS_PER_BLOCK =
    128; // ensure this matches policy expectations

// We'll pick simple alignment values:
constexpr int ALIGN_A = 8;
constexpr int ALIGN_B = 8;
constexpr int ALIGN_C = 8;

// Use a simple epilogue that computes: C = alpha * (A*B) + beta * C
using EpilogueOp =
    cutlass::gemm::blas_scaled_epilogue<ElementAccum, ElementAccum, ElementC>;

// --- Policy / block_task typedefs ---
// Note: API names vary slightly across CUTLASS versions. If your CUTLASS
// doesn't expose this exact type, search for cutlass::gemm::block_task and the
// policy helpers in your version. The block_task template below follows the
// classic CUTLASS pattern.
using block_task_policy_t = cutlass::gemm::block_task_policy<
    BLOCK_M, BLOCK_N, 8,
    4, // thread tile (ThreadItemsY, ThreadItemsX) — example numbers
    BLOCK_K,
    true, // double-buffering / double-scratch (policy detail)
    cutlass::gemm::block_raster_enum::Default>;

using block_task_t = cutlass::gemm::block_task<
    block_task_policy_t, ElementA, ElementAccum,
    cutlass::matrix_transform_t::NonTranspose, ALIGN_A,
    cutlass::matrix_transform_t::NonTranspose, ALIGN_B, EpilogueOp, ALIGN_C,
    true // ragged tiles allowed
    >;

// Device-callable wrapper that runs a single CUTLASS threadblock-level GEMM.
// smem_ptr must point to a per-CTA __shared__ allocation of
// block_task_t::scratch_storage_t. A_tile/B_tile/C_tile must already be
// pointers to the tile that *this* CTA will compute. lda/ldb/ldc are strides
// (in elements) for the input matrices.
__device__ __forceinline__ void
cutlass_gemm_device(ElementC *C_tile, ElementA const *A_tile,
                    ElementB const *B_tile, int lda, int ldb, int ldc,
                    int M_tile, int N_tile, int K_tile, float alpha, float beta,
                    typename block_task_t::scratch_storage_t *smem_ptr) {
  // Construct epilogue functor (device-callable)
  EpilogueOp epilogue(alpha, beta);

  // Construct block task object: signature depends on CUTLASS version; this
  // matches the common pattern:
  block_task_t task(reinterpret_cast<void *>(smem_ptr), smem_ptr, A_tile,
                    B_tile, C_tile, lda, ldb,
                    ldc, // some CUTLASS block_task constructors expect strides;
                         // if not, remove
                    epilogue, M_tile, N_tile, K_tile);

  // Execute the block's cooperative work. All threads in the CTA must call
  // this.
  task.run();
}

// A simple kernel that maps each CTA to a tile of C (tile size = BLOCK_M x
// BLOCK_N). For clarity this kernel uses a naive mapping: blockIdx.x =
// tile_row, blockIdx.y = tile_col. In a production mapping you would use
// swizzles and bounds checking.
__global__ void gemm_with_cutlass_device(ElementC *C, ElementA const *A,
                                         ElementB const *B, int M, int N, int K,
                                         int lda, int ldb, int ldc, float alpha,
                                         float beta) {
  // Each CTA needs its shared scratch storage (type provided by CUTLASS)
  __shared__ typename block_task_t::scratch_storage_t smem;

  // compute tile coordinates (row, col)
  int tile_row = blockIdx.x;
  int tile_col = blockIdx.y;

  // offsets (assume row-major layout for this example)
  int row_offset = tile_row * BLOCK_M;
  int col_offset = tile_col * BLOCK_N;

  // Compute pointer to A_tile, B_tile, C_tile for this CTA (note: do bounds
  // checking)
  ElementA const *A_tile =
      A + row_offset * lda; // A[row_offset : row_offset+BLOCK_M, :]
  ElementB const *B_tile =
      B + col_offset; // B[:, col_offset : col_offset+BLOCK_N] (if row-major B
                      // is layout dependent)
  ElementC *C_tile = C + row_offset * ldc + col_offset;

  // Compute effective tile dims (handle edge / ragged tiles)
  int M_tile = min(BLOCK_M, M - row_offset);
  int N_tile = min(BLOCK_N, N - col_offset);
  int K_tile =
      K; // we assume full K; for paneling you'd adjust this per-k-panel.

  // All threads in CTA must call the device wrapper.
  cutlass_gemm_device(C_tile, A_tile, B_tile, lda, ldb, ldc, M_tile, N_tile,
                      K_tile, alpha, beta, &smem);
}

// -------------------- Minimal host-side test / launcher --------------------
int main() {
  // Small example: M x K  *  K x N = M x N
  int M = 256;
  int N = 256;
  int K = 512;

  int lda = K;
  int ldb = N;
  int ldc = N;

  size_t size_A = size_t(M) * size_t(K);
  size_t size_B = size_t(K) * size_t(N);
  size_t size_C = size_t(M) * size_t(N);

  // Allocate device buffers
  ElementA *d_A;
  ElementB *d_B;
  ElementC *d_C;
  cudaMalloc(&d_A, size_A * sizeof(ElementA));
  cudaMalloc(&d_B, size_B * sizeof(ElementB));
  cudaMalloc(&d_C, size_C * sizeof(ElementC));

  // (Fill A/B on host and copy to device... omitted for brevity)
  // For testing you may cudaMemset the device arrays or upload simple patterns.

  dim3 grid((M + BLOCK_M - 1) / BLOCK_M, (N + BLOCK_N - 1) / BLOCK_N);
  dim3 block(THREADS_PER_BLOCK);

  float alpha = 1.0f, beta = 0.0f;
  gemm_with_cutlass_device<<<grid, block>>>(d_C, d_A, d_B, M, N, K, lda, ldb,
                                            ldc, alpha, beta);

  cudaDeviceSynchronize();
  printf("Launched device GEMM grid (%d,%d) with blockDim %d\n", grid.x, grid.y,
         block.x);

  // Free
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return 0;
}
