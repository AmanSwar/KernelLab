#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_intrinsics.h>
#include <__clang_cuda_runtime_wrapper.h>
#include <cuda_runtime.h>



__global__
void naive_bfs(
    int* adjacency,
    int* offset,
    int* visited,
    int* frontier,
    int* newFrontierSize,
    int frontierSize,
    int level
){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid < frontierSize){
        int vertex = frontier[tid];

        int start = offset[vertex];
        int end = offset[vertex + 1];

        for(int edge = start ; edge < end ; edge++){
            int neightbor = adjacency[edge];

            if(visited[neightbor] == -1){
                visited[neightbor] = level;
                int index = atomicAdd(newFrontierSize , 1);

            }
        }

    }
}


void launch_naive_bfs(
    int* adjacency,
    int* offset,
    int* visited,
    int* frontier,
    int* newFrontierSize,
    int frontierSize,
    int level
){

    int blockSize = 256;
    int gridSize = (frontierSize + blockSize -1) / blockSize;

    naive_bfs<<<gridSize , blockSize>>>(adjacency , offset , visited , frontier , newFrontierSize , frontierSize , level);
    cudaDeviceSynchronize();
}

