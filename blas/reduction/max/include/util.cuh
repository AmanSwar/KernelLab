#include <cuda_runtime.h>


#define WARP_SIZE 32

template <const int warp_size = WARP_SIZE>
__device__ float warp_reduce_max(float val) {
  unsigned int mask = 0xFFFFFFFF;

  for (int offset = WARP_SIZE >> 1; offset >= 1; offset >>= 1) {
    val = max(val, __shfl_xor_sync(mask, val, offset));
  }

  return val;
}




