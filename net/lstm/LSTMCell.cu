#include <hip/hip_runtime.h>


#define    kColumsPerBlock   64
#define    kThreadNumPerBlock 256
#define    kHiddenSize 256
#define    kInputSize 256

#define sigmod(x)  1.000000e+00f / (1.000000e+00f + __expf(0.000000e+00f - x))


extern "C" __global__ void gemv(const float *__restrict__ input,
                     const float *__restrict__ weight,
                     float *__restrict__ output, int block_offset)
{
    __shared__ float nndense_output[kColumsPerBlock];
    const int warp_id = threadIdx.x >> 6;
    const int lane_id = threadIdx.x & 0x3f;
    const int colOffset = (blockIdx.x + block_offset) * kColumsPerBlock + lane_id;
    nndense_output[lane_id] = 0.0000f;
    float temp = 0.0000f;
    const int ROWS = kInputSize / (kThreadNumPerBlock >> 6);
    int vectorRow = ROWS * warp_id;
    int kStart =
        vectorRow * kHiddenSize + (blockIdx.x + block_offset) * kColumsPerBlock + lane_id;
    int kEnd = kStart + ROWS * kHiddenSize;
    for (; kStart < kEnd; kStart += kHiddenSize, ++vectorRow)
    {
        const float data = input[vectorRow];
        temp = fma(weight[kStart], data, temp);
    }

    atomicAdd(&nndense_output[lane_id], temp);
    __syncthreads();
    if (warp_id == 0)
        output[colOffset] = nndense_output[lane_id];
}

extern "C" __global__ void solve(float *t00, float *t01, float *b0, float *t10, float *t11,
                      float *b1, float *t20, float *t21, float *b2, float *t30,
                      float *t31, float *b3, float *state_c, float *state_h)
{
    const int idx = threadIdx.x;
    float x = t00[idx] + t01[idx] + b0[idx];
    float y = t10[idx] + t11[idx] + b1[idx];
    float z = t20[idx] + t21[idx] + b2[idx];
    float w = t30[idx] + t31[idx] + b3[idx];
    x = sigmoid(x);
    y = tanh(y);
    w = sigmoid(w);
    z = sigmoid(z + 1.0000f) * state_c[idx];
    state_c[idx] = fma(x, y, z);
    state_h[idx] = (tanh(state_c[idx])) * w;
    // sigmoid(z) + 1.0000f
}
