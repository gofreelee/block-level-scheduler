#include "../model.h"
#include <iostream>
int main()
{
    std::string test_file_name = "toy.cu";
    model  toy_model(test_file_name);
    auto kernels = toy_model.get_kernel_functions();
    for(auto func:  kernels)
    {
        std::cout << func << std::endl;
        std::cout << "****************************" << std::endl;
    }
}