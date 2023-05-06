#pragma once
#include <queue>
#include <vector>
#include <string>
#include <map>
#include <hip/hip_runtime.h>
class model{
    std::string model_name;    
    std::string model_file_path;
    std::vector<std::string>  kernel_source_list;
    std::vector<std::pair<dim3, dim3>> kernel_launch_params;
    std::vector<int> current_have_launched_info;  // index is which kernel,  corresponding value is complete blocks number;
    int kernel_offset;
    //I need know every kernel information, launch blocksize threadsize
public:
    model(std::string kernel_file_path); //need to split source to  pieces of kernel functions;
    std::vector<std::string>  get_kernel_functions(){
        return kernel_source_list;
    } 
    
};