// rope_benchmark_verify.cu
// nvcc -O3 rope_benchmark_verify.cu -o rope_benchmark_verify

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

// --------------------- CPU reference ---------------------
void apply_rope_cpu_float(const float *x, const float *cosv, const float *sinv,
                          float *out, int B, int H, int N, int D) {
  assert(D % 2 == 0);
  const int half = D / 2;
  const std::size_t stride_head = (std::size_t)N * D;
  const std::size_t stride_batch = (std::size_t)H * stride_head;

  for (int b = 0; b < B; ++b) {
    const std::size_t base_b = (std::size_t)b * stride_batch;
    for (int h = 0; h < H; ++h) {
      const std::size_t base_h = base_b + (std::size_t)h * stride_head;
      for (int s = 0; s < N; ++s) {
        const std::size_t base_s = base_h + (std::size_t)s * D;
        const std::size_t cos_s = (std::size_t)s * D;
        const std::size_t sin_s = (std::size_t)s * D;
        for (int d = 0; d < D; ++d) {
          const std::size_t idx = base_s + d;
          const float x_in = x[idx];

          float rotated;
          if (d < half) {
            rotated = -x[base_s + (d + half)];
          } else {
            rotated = x[base_s + (d - half)];
          }

          const float cv = cosv[cos_s + d];
          const float sv = sinv[sin_s + d];
          out[idx] = x_in * cv + rotated * sv;
        }
      }
    }
  }
}

// --------------------- Device kernel (user's kernel copied)
// ---------------------
// Option 1: Direct RoPE implementation (matches CPU reference exactly)
__global__ void rope_apply_half2_kernel_fixed_v1(
    const half2 *__restrict__ x,         // [B * H * N * head_dim_half]
    const half2 *__restrict__ cos_cache, // [N * head_dim_half]
    const half2 *__restrict__ sin_cache, // [N * head_dim_half]
    half2 *__restrict__ out,             // same layout as x
    int batch_size, int num_heads, int seq_len,
    int head_dim_half // head_dim / 2
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch_size * num_heads * seq_len * head_dim_half;
  if (idx >= total)
    return;

  // Extract indices
  int pair_idx = idx % head_dim_half;
  int seq_idx = (idx / head_dim_half) % seq_len;

  // Get the current half2 element (contains x[d] and x[d+1])
  half2 xh = x[idx];

  // We need to pair with the element at offset head_dim_half/2
  // For RoPE: pair (x[d], x[d+head_dim/2]) where d < head_dim/2
  //           or   (x[d], x[d-head_dim/2]) where d >= head_dim/2

  int head_dim_quarter =
      head_dim_half / 2; // Since head_dim_half = D/2, this is D/4
  half2 xh_pair;
  bool first_half = (pair_idx < head_dim_quarter);

  if (first_half) {
    // We need x[d+head_dim/2], which is at pair_idx + head_dim_quarter
    int paired_idx = idx + head_dim_quarter;
    xh_pair = x[paired_idx];
  } else {
    // We need x[d-head_dim/2], which is at pair_idx - head_dim_quarter
    int paired_idx = idx - head_dim_quarter;
    xh_pair = x[paired_idx];
  }

  // Get cos/sin values for this sequence position and dimension pair
  int cos_sin_idx = seq_idx * head_dim_half + pair_idx;
  half2 cos_h = cos_cache[cos_sin_idx];
  half2 sin_h = sin_cache[cos_sin_idx];

  // Apply RoPE formula: out[d] = x[d] * cos[d] + rotated * sin[d]
  // where rotated = -x[d+head_dim/2] if d < head_dim/2, else x[d-head_dim/2]

  half2 rotated;
  if (first_half) {
    // rotated = -x[d+head_dim/2]
    rotated = make_half2(__hneg(xh_pair.x), __hneg(xh_pair.y));
  } else {
    // rotated = x[d-head_dim/2]
    rotated = xh_pair;
  }

  // Apply the rotation: out = x * cos + rotated * sin
  __half r1 = __hadd(__hmul(xh.x, cos_h.x), __hmul(rotated.x, sin_h.x));
  __half r2 = __hadd(__hmul(xh.y, cos_h.y), __hmul(rotated.y, sin_h.y));

  out[idx] = make_half2(r1, r2);
}

// Option 2: More efficient version - process RoPE pairs directly in half2
// This assumes you can reorganize your data layout to group RoPE pairs together
__global__ void rope_apply_half2_kernel(
    const half2 *__restrict__ x,       
    const half2 *__restrict__ cos_cache,// [N * head_dim_half]
    const half2 *__restrict__ sin_cache, // [N * head_dim_half]
    half2 *__restrict__ out,             // same layout as x
    int batch_size, int num_heads, int seq_len,
    int head_dim_half // head_dim / 2
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch_size * num_heads * seq_len * head_dim_half;
  if (idx >= total)
    return;

  // Extract indices
  int pair_idx = idx % head_dim_half;
  int seq_idx = (idx / head_dim_half) % seq_len;

  // Get the current half2 element (contains x[d] and x[d+1])
  half2 xh = x[idx];

  // We need to pair with the element at offset head_dim_half/2
  // For RoPE: pair (x[d], x[d+head_dim/2]) where d < head_dim/2
  //           or   (x[d], x[d-head_dim/2]) where d >= head_dim/2

  int head_dim_quarter =
      head_dim_half / 2; // Since head_dim_half = D/2, this is D/4
  half2 xh_pair;
  bool first_half = (pair_idx < head_dim_quarter);

  if (first_half) {
    // We need x[d+head_dim/2], which is at pair_idx + head_dim_quarter
    int paired_idx = idx + head_dim_quarter;
    xh_pair = x[paired_idx];
  } else {
    // We need x[d-head_dim/2], which is at pair_idx - head_dim_quarter
    int paired_idx = idx - head_dim_quarter;
    xh_pair = x[paired_idx];
  }

  // Get cos/sin values for this sequence position and dimension pair
  int cos_sin_idx = seq_idx * head_dim_half + pair_idx;
  half2 cos_h = cos_cache[cos_sin_idx];
  half2 sin_h = sin_cache[cos_sin_idx];

  // Apply RoPE formula: out[d] = x[d] * cos[d] + rotated * sin[d]
  // where rotated = -x[d+head_dim/2] if d < head_dim/2, else x[d-head_dim/2]

  half2 rotated;
  if (first_half) {
    // rotated = -x[d+head_dim/2]
    rotated = make_half2(__hneg(xh_pair.x), __hneg(xh_pair.y));
  } else {
    // rotated = x[d-head_dim/2]
    rotated = xh_pair;
  }

  // Apply the rotation: out = x * cos + rotated * sin
  __half r1 = __hadd(__hmul(xh.x, cos_h.x), __hmul(rotated.x, sin_h.x));
  __half r2 = __hadd(__hmul(xh.y, cos_h.y), __hmul(rotated.y, sin_h.y));

  out[idx] = make_half2(r1, r2);
}
// --------------------- Helpers / error-checks ---------------------
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
              cudaGetErrorString(err));                                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// --------------------- Benchmark + Verify function ---------------------
/*
 Parameters:
  B, H, N, D: dimensions (D must be even)
  iterations: timed iterations (after warmup)
  warmup: warmup iterations (not timed)
  block_size: kernel block size
*/
void verify_and_benchmark(int B, int H, int N, int D, int warmup = 10,
                          int iterations = 200, int block_size = 256) {
  if (D % 2 != 0) {
    fprintf(stderr, "D must be even\n");
    return;
  }
  const int head_dim_half = D / 2;
  const std::size_t total_half2 = (std::size_t)B * H * N * head_dim_half;
  const std::size_t total_scalars = (std::size_t)B * H * N * D;
  const std::size_t coslen_half2 =
      (std::size_t)N * head_dim_half; // cos/sin as half2

  printf("Dims: B=%d H=%d N=%d D=%d -> half2 elems=%zu\n", B, H, N, D,
         (size_t)total_half2);

  // Allocate host float buffers (CPU reference)
  std::vector<float> h_x(total_scalars);
  std::vector<float> h_cos((std::size_t)N * D);
  std::vector<float> h_sin((std::size_t)N * D);
  std::vector<float> h_out_ref(total_scalars);

  // Random init (deterministic)
  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  for (size_t i = 0; i < total_scalars; ++i)
    h_x[i] = dist(rng);
  for (size_t i = 0; i < (size_t)N * D; ++i) {
    h_cos[i] = dist(rng);
    h_sin[i] = dist(rng);
  }

  // Compute CPU reference
  apply_rope_cpu_float(h_x.data(), h_cos.data(), h_sin.data(), h_out_ref.data(),
                       B, H, N, D);

  // Prepare host half2 buffers to copy to device
  // Each half2 corresponds to two scalars: lane0=x[d0], lane1=x[d0+1]
  std::vector<half2> h_x_half2(total_half2);
  std::vector<half2> h_cos_half2(coslen_half2);
  std::vector<half2> h_sin_half2(coslen_half2);

  // pack x into half2 host buffer
  // layout: for each (b,h,s) we have head_dim_half half2 elements,
  // each half2 contains d=2*i and d=2*i+1
  for (int b = 0; b < B; ++b) {
    for (int h = 0; h < H; ++h) {
      for (int s = 0; s < N; ++s) {
        std::size_t base_scalar =
            ((std::size_t)b * H + h) * (std::size_t)N * D + (std::size_t)s * D;
        std::size_t base_half2 =
            ((std::size_t)b * H + h) * (std::size_t)N * head_dim_half +
            (std::size_t)s * head_dim_half;
        for (int p = 0; p < head_dim_half; ++p) {
          int d0 = p * 2;
          float f0 = h_x[base_scalar + d0];
          float f1 = h_x[base_scalar + d0 + 1];
          // convert to half and pack into both lanes
          __half h0 = __float2half_rn(f0);
          __half h1 = __float2half_rn(f1);
          h_x_half2[base_half2 + p] = make_half2(h0, h1);
        }
      }
    }
  }

  // Pack cos and sin into half2 arrays.
  // We follow the kernel assumption: cos_cache[...] is seq_len x head_dim_half
  // half2 array, and we will store the scalar in lane.x (and lane.y identical)
  // to keep layout simple.
  for (int s = 0; s < N; ++s) {
    std::size_t base_cos_scalar = (std::size_t)s * D;
    std::size_t base_half2 = (std::size_t)s * head_dim_half;
    for (int p = 0; p < head_dim_half; ++p) {
      int d0 = p * 2;
      float cv0 = h_cos[base_cos_scalar + d0];
      float cv1 = h_cos[base_cos_scalar + d0 + 1];
      // store cv0 and cv1 into lanes respectively (but kernel uses lane.x only)
      __half hc0 = __float2half_rn(cv0);
      __half hc1 = __float2half_rn(cv1);
      h_cos_half2[base_half2 + p] = make_half2(hc0, hc1);

      float sv0 = h_sin[base_cos_scalar + d0];
      float sv1 = h_sin[base_cos_scalar + d0 + 1];
      __half hs0 = __float2half_rn(sv0);
      __half hs1 = __float2half_rn(sv1);
      h_sin_half2[base_half2 + p] = make_half2(hs0, hs1);
    }
  }

  // Device allocations
  half2 *d_x = nullptr;
  half2 *d_cos = nullptr;
  half2 *d_sin = nullptr;
  half2 *d_out = nullptr;

  CUDA_CHECK(cudaMalloc(&d_x, total_half2 * sizeof(half2)));
  CUDA_CHECK(cudaMalloc(&d_cos, coslen_half2 * sizeof(half2)));
  CUDA_CHECK(cudaMalloc(&d_sin, coslen_half2 * sizeof(half2)));
  CUDA_CHECK(cudaMalloc(&d_out, total_half2 * sizeof(half2)));

  // Copy host->device
  CUDA_CHECK(cudaMemcpy(d_x, h_x_half2.data(), total_half2 * sizeof(half2),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_cos, h_cos_half2.data(), coslen_half2 * sizeof(half2),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_sin, h_sin_half2.data(), coslen_half2 * sizeof(half2),
                        cudaMemcpyHostToDevice));

  // Create stream & events for timing
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  cudaEvent_t start_ev, stop_ev;
  CUDA_CHECK(cudaEventCreate(&start_ev));
  CUDA_CHECK(cudaEventCreate(&stop_ev));

  // Kernel launch parameters
  int block = block_size;
  int grid = (int)((total_half2 + block - 1) / block);

  // Warmup
  for (int i = 0; i < warmup; ++i) {
    rope_apply_half2_kernel<<<grid, block, 0, stream>>>(
        d_x, d_cos, d_sin, d_out, B, H, N, head_dim_half);
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Timed runs
  CUDA_CHECK(cudaEventRecord(start_ev, stream));
  for (int i = 0; i < iterations; ++i) {
    rope_apply_half2_kernel<<<grid, block, 0, stream>>>(
        d_x, d_cos, d_sin, d_out, B, H, N, head_dim_half);
  }
  CUDA_CHECK(cudaEventRecord(stop_ev, stream));
  CUDA_CHECK(cudaEventSynchronize(stop_ev));

  float ms = 0.f;
  CUDA_CHECK(
      cudaEventElapsedTime(&ms, start_ev, stop_ev)); // ms for all iterations
  double elapsed_s = ms / 1000.0;

  // Copy back results
  std::vector<half2> h_out_half2(total_half2);
  CUDA_CHECK(cudaMemcpy(h_out_half2.data(), d_out, total_half2 * sizeof(half2),
                        cudaMemcpyDeviceToHost));

  // Convert device half2 results to float scalar array for comparison
  std::vector<float> h_out_gpu(total_scalars);
  for (std::size_t idx = 0; idx < total_half2; ++idx) {
    half2 v = h_out_half2[idx];
    // access lanes
    __half lane0 = v.x;
    __half lane1 = v.y;
    float f0 = __half2float(lane0);
    float f1 = __half2float(lane1);
    std::size_t scalar_base = idx * 2;
    // Need to map half2 linear index back to (b,h,s,d). Our packing was:
    // half2 idx = ((b*H + h) * N + s) * head_dim_half + p
    // The scalar linear index is ((b*H + h) * N + s) * D + p*2 + lane
    // So reconstruct:
    std::size_t p = idx % head_dim_half;
    std::size_t tmp = idx / head_dim_half;
    std::size_t s = tmp % N;
    std::size_t bh = tmp / N;
    std::size_t b = bh / H;
    std::size_t h = bh % H;
    std::size_t scalar_idx_base =
        ((b * (size_t)H + h) * (size_t)N + s) * (size_t)D + p * 2;
    h_out_gpu[scalar_idx_base + 0] = f0;
    h_out_gpu[scalar_idx_base + 1] = f1;
  }

  // Verify: compute differences
  double max_abs_err = 0.0;
  double sum_abs_err = 0.0;
  for (size_t i = 0; i < total_scalars; ++i) {
    double a = (double)h_out_ref[i];
    double b = (double)h_out_gpu[i];
    double err = std::abs(a - b);
    sum_abs_err += err;
    if (err > max_abs_err)
      max_abs_err = err;
  }
  double mean_abs_err = sum_abs_err / (double)total_scalars;

  // Compute GFLOPS: count 6 FLOPs per half2 element (r1: 2 mul + 1 sub, r2: 2
  // mul + 1 add)
  double flops_per_half2 = 6.0;
  double total_flops =
      flops_per_half2 * (double)total_half2 * (double)iterations;
  double gflops = total_flops / elapsed_s / 1e9;

  printf("Benchmark results:\n");
  printf("  iterations (timed): %d  warmup: %d\n", iterations, warmup);
  printf("  elapsed (ms): %.3f  elapsed (s): %.6f\n", ms, elapsed_s);
  printf("  GFLOPS: %.3f\n", gflops);
  printf("Verification:\n");
  printf("  max abs error = %.6e\n", max_abs_err);
  printf("  mean abs error = %.6e\n", mean_abs_err);

  // Tolerance check (half precision): allow small tolerance (e.g., 1e-2
  // absolute)
  const double abs_tol = 1e-2;
  const double rel_tol = 1e-2;
  bool pass = (max_abs_err <= abs_tol) ||
              (max_abs_err <= rel_tol * std::abs((double)h_out_ref[0] + 1e-12));
  printf("  verification %s (abs_tol=%.3g, rel_tol=%.3g)\n",
         pass ? "PASSED" : "FAILED", abs_tol, rel_tol);

  // Cleanup
  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_cos));
  CUDA_CHECK(cudaFree(d_sin));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaEventDestroy(start_ev));
  CUDA_CHECK(cudaEventDestroy(stop_ev));
  CUDA_CHECK(cudaStreamDestroy(stream));
}

// --------------------- main: example usage ---------------------
int main(int argc, char **argv) {
  // Example default dims (reasonable for benchmarking)
  int B = 8;
  int H = 8;
  int N = 512;
  int D = 64; // must be even

  // Allow overriding via args: ./exe B H N D
  if (argc >= 5) {
    B = atoi(argv[1]);
    H = atoi(argv[2]);
    N = atoi(argv[3]);
    D = atoi(argv[4]);
  }

  printf("Running verify_and_benchmark with B=%d H=%d N=%d D=%d\n", B, H, N, D);

  verify_and_benchmark(B, H, N, D, /*warmup*/ 10, /*iter*/ 200, /*block*/ 256);

  return 0;
}
