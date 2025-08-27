#include <cstdio>
#include <cuda_fp16.h>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/epilogue.h"
#include "cutlass/gemm/block_task.h" // threadblock-level block_task (older / stable pattern)
#include "cutlass/gemm/gemm.h"
#include "cutlass/half.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/matrix_ref.h"

using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = cutlass::half_t;
using ElementAccumulator = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

static int const kThreadblockShapeM = 128;
static int const kThreadblockShapeN = 256;
static int const kThreadblockShapeK = 32;

static int const kWarpShapeM = 64;
static int const kWarpShapeN = 64;
static int const kWarpShapeK = 32;

static int const kInstructionShapeM = 16;
static int const kInstructionShapeN = 8;
static int const kInstructionShapeK = 8;

static int const kStages = 2;

using Gemm = cutlass::gemm::kernel::DefaultGemm<
    ElementA, LayoutA, cutlass::ComplexTransform::kNone, 8, ElementB, LayoutB,
    cutlass::ComplexTransform::kNone, 8, ElementC, LayoutC, ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75, // Use Sm80 for A100, Sm75 for V100/T4
    cutlass::gemm::GemmShape<kThreadblockShapeM, kThreadblockShapeN,
                             kThreadblockShapeK>,
    cutlass::gemm::GemmShape<kWarpShapeM, kWarpShapeN, kWarpShapeK>,
    cutlass::gemm::GemmShape<kInstructionShapeM, kInstructionShapeN,
                             kInstructionShapeK>,
    cutlass::epilogue::thread::LinearCombination<
        ElementC, 128 / cutlass::sizeof_bits<ElementC>::value,
        ElementAccumulator, ElementAccumulator>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, kStages,
    true, // SplitKSerial
    cutlass::arch::OpMultiplyAdd>;

__device__ void cutlass_gemm_device(int M, int N, int K, ElementA const *A,
                                    int lda, ElementB const *B, int ldb,
                                    ElementC *C, int ldc,
                                    ElementAccumulator alpha = 1.0f,
                                    ElementAccumulator beta = 0.0f) {
  // GEMM problem size
  cutlass::gemm::GemmCoord problem_size(M, N, K);

  // Create GEMM arguments
  typename Gemm::Arguments arguments{problem_size, {A, lda}, {B, ldb},
                                     {C, ldc},     {C, ldc}, {alpha, beta}};

  // Get shared memory size
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate shared memory (this must be done carefully in device code)
  extern __shared__ uint8_t shared_memory[];

  // Initialize GEMM operator
  Gemm gemm_op;

  // Check if arguments are valid
  cutlass::Status status = gemm_op.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    return; // Handle error appropriately
  }

  // Initialize the GEMM operator
  status = gemm_op.initialize(arguments, shared_memory);
  if (status != cutlass::Status::kSuccess) {
    return; // Handle error appropriately
  }

  // Execute the GEMM
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) {
    return; // Handle error appropriately
  }
}
