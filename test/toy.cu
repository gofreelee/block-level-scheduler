#include <hip/hip_runtime.h>

extern "C" __global__ void vecadd(float *A, float *B, float *C)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    C[idx] = A[idx] + B[idx];
}

extern "C" __global__ void proxy_kernel(float *A, float *B, float *C, float *A_, float *B_, float *C_, int be_block_offset)
{
    if (blockIdx.x < 32)
    {
        vecadd(A, B, C);
    }
    else
    {
        //printf("debug\n");
        vecadd_be(A_, B_, C_, 32, be_block_offset);
    }
}

extern "C" __global__ void vecadd_be(float *A, float *B, float *C, int start_cu_offset, int be_block_offset)
{
    if ((blockIdx.x) >= start_cu_offset)
    {
        if ((blockIdx.x - start_cu_offset) + be_block_offset < 32)
        {
            int idx = (blockIdx.x - start_cu_offset+ be_block_offset) * blockDim.x + threadIdx.x;
            C[idx] = A[idx] + B[idx];
        }
    }
    // if(blockIdx.x >= 31){
    //     printf("vecadd_be\n");
    // }
}