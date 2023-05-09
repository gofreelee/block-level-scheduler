#include "LSTMCell.h"
#include <iostream>

extern "C" void gemv(const float *__restrict__ input, const float *__restrict__ weight,
                     float *__restrict__ output);

extern "C" void solve(float *t00, float *t01, float *b0, float *t10, float *t11, float *b1,
                      float *t20, float *t21, float *b2, float *t30, float *t31, float *b3,
                      float *state_c, float *state_h);

void LSTMCell::init(const HostCellParams &params)
{
    for (int i = 0; i < 8; ++i)
        hipMalloc(&tmp_outputs[i], sizeof(float) * hidden_size);
    hipMalloc(&state_c_dev, sizeof(float) * (4 * hidden_size * input_size +
                                             4 * hidden_size * hidden_size +
                                             6 * hidden_size));

    output_host = (float *)malloc(sizeof(float) * hidden_size);

    state_h_dev = state_c_dev + hidden_size;
    for (int i = 0; i < 4; ++i)
    {
        W_dev[i] = state_h_dev + hidden_size + i * hidden_size * input_size;
        U_dev[i] = state_h_dev + hidden_size + 4 * hidden_size * input_size +
                   i * hidden_size * hidden_size;
        bias_dev[i] = state_h_dev + hidden_size + 4 * hidden_size * input_size +
                      4 * hidden_size * hidden_size + i * hidden_size;
    }

    hipMemcpy(state_c_dev, params.init_state_c, sizeof(float) * hidden_size,
              hipMemcpyHostToDevice);
    hipMemcpy(state_c_dev, params.init_state_c, sizeof(float) * hidden_size,
              hipMemcpyHostToDevice);
    hipMemcpy(state_h_dev, params.init_state_h, sizeof(float) * hidden_size,
              hipMemcpyHostToDevice);
    hipMemcpy(W_dev[0], params.W, sizeof(float) * hidden_size * input_size * 4,
              hipMemcpyHostToDevice);
    hipMemcpy(U_dev[0], params.U,
              sizeof(float) * hidden_size * hidden_size * 4,
              hipMemcpyHostToDevice);
    hipMemcpy(bias_dev[0], params.bias, sizeof(float) * hidden_size * 4,
              hipMemcpyHostToDevice);

    void **WI_0 = (void **)malloc(sizeof(void *) * 4);
    void **WI_1 = (void **)malloc(sizeof(void *) * 4);
    void **WI_2 = (void **)malloc(sizeof(void *) * 4);
    void **WI_3 = (void **)malloc(sizeof(void *) * 4);
    void **UH_0 = (void **)malloc(sizeof(void *) * 4);
    void **UH_1 = (void **)malloc(sizeof(void *) * 4);
    void **UH_2 = (void **)malloc(sizeof(void *) * 4);
    void **UH_3 = (void **)malloc(sizeof(void *) * 4);
    WI_0[0] = &input_dev;
    WI_0[1] = &W_dev[0];
    WI_0[2] = &tmp_outputs[0];

    WI_1[0] = &input_dev;
    WI_1[1] = &W_dev[1];
    WI_1[2] = &tmp_outputs[1];
    
    WI_2[0] = &input_dev;
    WI_2[1] = &W_dev[2];
    WI_2[2] = &tmp_outputs[2];

    WI_3[0] = &input_dev;
    WI_3[1] = &W_dev[3];
    WI_3[2] = &tmp_outputs[3];

    UH_0[0] = &input_dev;
    UH_0[1] = &U_dev[0];
    UH_0[2] = &tmp_outputs[4];

    UH_1[0] = &input_dev;
    UH_1[1] = &U_dev[1];
    UH_1[2] = &tmp_outputs[5];

    UH_2[0] = &input_dev;
    UH_2[1] = &U_dev[2];
    UH_2[2] = &tmp_outputs[6];
    
    UH_3[0] = &input_dev;
    UH_3[1] = &U_dev[2];
    UH_3[2] = &tmp_outputs[7];

    this->kernel_launch_params.push_back(WI_0);
    this->kernel_launch_params.push_back(WI_1);
    this->kernel_launch_params.push_back(WI_2);
    this->kernel_launch_params.push_back(WI_3);
    this->kernel_launch_params.push_back(UH_0);
    this->kernel_launch_params.push_back(UH_1);
    this->kernel_launch_params.push_back(UH_2);
    this->kernel_launch_params.push_back(UH_3);

    hipStreamCreate(&stream_t);
}

void LSTMCell::compute(float *input_dev)
    // {
    //     float *tmp_outputs[8];
    //     for (int i = 0; i < 8; ++i)
    //         hipMalloc(&tmp_outputs[i], sizeof(float) * hidden_size);
    void *WI_0[] = {&input_dev, &W_dev[0], &tmp_outputs[0]};
void *WI_1[] = {&input_dev, &W_dev[1], &tmp_outputs[1]};
void *WI_2[] = {&input_dev, &W_dev[2], &tmp_outputs[2]};
void *WI_3[] = {&input_dev, &W_dev[3], &tmp_outputs[3]};
void *UH_0[] = {&state_h_dev, &U_dev[0], &tmp_outputs[4]};
void *UH_1[] = {&state_h_dev, &U_dev[1], &tmp_outputs[5]};
void *UH_2[] = {&state_h_dev, &U_dev[2], &tmp_outputs[6]};
void *UH_3[] = {&state_h_dev, &U_dev[3], &tmp_outputs[7]};

hipLaunchKernel((const void *)gemv, dim3(hidden_size >> 6), dim3(256),
                (void **)WI_0, COLUMNS_PER_BLOCK * sizeof(float),
                stream_t);
hipLaunchKernel((const void *)gemv, dim3(hidden_size >> 6), dim3(256),
                (void **)WI_1, COLUMNS_PER_BLOCK * sizeof(float),
                stream_t);
hipLaunchKernel((const void *)gemv, dim3(hidden_size >> 6), dim3(256),
                (void **)WI_2, COLUMNS_PER_BLOCK * sizeof(float),
                stream_t);
hipLaunchKernel((const void *)gemv, dim3(hidden_size >> 6), dim3(256),
                (void **)WI_3, COLUMNS_PER_BLOCK * sizeof(float),
                stream_t);
hipLaunchKernel((const void *)gemv, dim3(hidden_size >> 6), dim3(256),
                (void **)UH_0, COLUMNS_PER_BLOCK * sizeof(float),
                stream_t);
hipLaunchKernel((const void *)gemv, dim3(hidden_size >> 6), dim3(256),
                (void **)UH_1, COLUMNS_PER_BLOCK * sizeof(float),
                stream_t);
hipLaunchKernel((const void *)gemv, dim3(hidden_size >> 6), dim3(256),
                (void **)UH_2, COLUMNS_PER_BLOCK * sizeof(float),
                stream_t);
hipLaunchKernel((const void *)gemv, dim3(hidden_size >> 6), dim3(256),
                (void **)UH_3, COLUMNS_PER_BLOCK * sizeof(float),
                stream_t);
void *solve_args[] = {&tmp_outputs[0], &tmp_outputs[4], &bias_dev[0],
                      &tmp_outputs[1], &tmp_outputs[5], &bias_dev[1],
                      &tmp_outputs[2], &tmp_outputs[6], &bias_dev[2],
                      &tmp_outputs[3], &tmp_outputs[7], &bias_dev[3],
                      &state_c_dev, &state_h_dev};

hipLaunchKernel((const void *)solve, dim3(1), dim3(hidden_size),
                (void **)solve_args, 0, stream_t);
hipDeviceSynchronize();
for (int i = 0; i < 8; ++i)
    hipFree(tmp_outputs[i]);
}

void LSTMCell::Close()
{
    free(output_host);
    hipFree(state_c_dev);
    hipStreamDestroy(stream_t);
}

void LSTMCell::launch_kernel_with_blocknum(int kernel_index, int block_complete_number)
{
    // get what kernel should be launch
    auto kernel_sources = this->get_kernel_functions();
    auto kernel_name = this->get_kernel_launch_order()[kernel_index];

    // I should get the kernel parameters
    void *args[] = this->kernel_launch_params[kernel_index];

    hipFunction_t function;
    hipModuleGetFunction(&function, net_module, kernel_name);

    int gridSize = this->kernel_grid_block_params[kernel_index].first;
    int blockSize = this->kernel_grid_block_params[kernel_index].second;
    args[3] = &block_complete_number;

    hipModuleLaunchKernel(function, gridSize, 1, 1, blockSize, 1, 1, 0, 0, args, NULL);
    
}