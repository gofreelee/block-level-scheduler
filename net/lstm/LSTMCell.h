#pragma once
#include "hip/hip_runtime.h"
#include "../../model.h"

struct HostCellParams
{
    const float *init_state_c;
    const float *init_state_h;
    const float *W;
    const float *U;
    const float *bias;
};

class LSTMCell : public model
{

public:
    friend class LSTMNet;

    LSTMCell(size_t inputSize, size_t hiddenSize)
        : input_size(inputSize), hidden_size(hiddenSize)
    {
        set_kernel_launch_order(std::vector<std::string>{
            "gemv",
            "gemv",
            "gemv",
            "gemv",
            "gemv",
            "gemv",
            "gemv",
            "gemv",
            "solve"
        });
    }

    void init(const HostCellParams &param);

    void compute(float *input_dev);

    float *getResult()
    {
        hipMemcpy(output_host, state_h_dev, sizeof(float) * hidden_size,
                  hipMemcpyDeviceToHost);
        return output_host;
    }

    void Close();

    void launch_kernel_with_blocknum(int kernel_index, int block_complete_number) override;

private:
    size_t input_size, hidden_size;
    float *W_dev[3], *U_dev[4], *bias_dev[4];
    float *state_h_dev;
    float *state_c_dev;
    float *input_dev;
    float *output_host;

    float *tmp_outputs[8];
    hipStream_t stream_t;
};