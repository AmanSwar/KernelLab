#include <cfloat>
#include <cutlass/arch/arch.h>
#include <cutlass/arch/mma.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/kernel/default_gemm.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
#include <cutlass/gemm/warp/default_mma_tensor_op.h>
#include <cutlass/gemm_coord.h>
#include <cutlass/half.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>


using Query_dtype = cutlass::half_t;
using Key_dtype = cutlass::half_t;
using Score_dtype = cutlass::half_t; // score = gemm(Q,K)
using Acc_dtype = cutlass::half_t;


using QueryLayout = cutlass::layout::RowMajor;
using KeyLayout  = cutlass::layout::RowMajor;
using ScoreLayout = cutlass::layout::RowMajor;

//QK gemm
using QKGemm = cutlass::gemm::device::Gemm<
    Query_dtype, QueryLayout,
    Key_dtype , KeyLayout,
    Score_dtype , ScoreLayout,
    Acc_dtype,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128 , 128 , 32>, // thread block shape
    cutlass::gemm::GemmShape<64 , 64 , 32>, // Warp Shape
    cutlass::gemm::GemmShape<16,8,8>, // instruction shape
    cutlass::epilogue::thread::LinearCombination<
        Score_dtype , 128 / cutlass::sizeof_bits<Score_dtype>::value,
        Acc_dtype , Acc_dtype>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2
    >; 


__global__ void gemm_kernel(
    Query_dtype const* Query,
    Key_dtype const* Key,
    Score_dtype * Score,
    int M , int N , int K,
    Acc_dtype alpha , Acc_dtype beta
){
    cutlass::gemm::GemmCoord problem_size(M , N , K);

    typename QKGemm::Arguments qk_gemm_args{
        problem_size ,
        {Query , K} , 
        {Key , N},
        {Score , N},
        {Score , N},
        {alpha , beta}
    };


    QKGemm qk_gemm_op;

    if(qk_gemm_op.can_implement(qk_gemm_args)){
        cutlass::Status status = qk_gemm_op.initialize(qk_gemm_args);

        if(status == cutlass::Status::kSuccess){
            status = qk_gemm_op();
        }
    }
}


void launch_cutlass_gemm(
    const cutlass::half_t *d_A,
    const cutlass::half_t *d_B,
    cutlass::half_t* d_C , 
    int M , int N , int K,
    cudaStream_t stream = 0
){
    dim3 grid(1,1);
    dim3 block(32 ,  32);

    cutlass::half_t alpha(1.0f);
    cutlass::half_t beta(0.0f);


    
}