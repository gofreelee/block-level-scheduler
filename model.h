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
    int kernel_offset;
    //I need know every kernel information, launch blocksize threadsize
public:

};