#pragma once
#include <queue>
#include <vector>
#include <string>
#include <map>
#include <hip/hip_runtime.h>
class model{

protected:
    std::string model_name;    
    std::string model_file_path;
    std::map<std::string, std::string>  kernel_source_list;
    std::vector<std::string>  kernel_launch_order;
    std::vector<void *[]> kernel_launch_params;
    std::vector<int> current_have_launched_info;  // index is which kernel,  corresponding value is complete blocks number;
    std::vector<std::pair<int, int>> kernel_grid_block_params;
    int kernel_offset;
    hipModule_t net_module;
    //I need know every kernel information, launch blocksize threadsize
public:
    model(std::string kernel_file_path); //need to split source to  pieces of kernel functions;
    std::map<std::string, std::string>  get_kernel_functions(){
        return kernel_source_list;
    } 

    void set_kernel_launch_order(std::vector<std::string> launch_order){
        this->kernel_launch_order = launch_order;
    }

    std::vector<std::string> get_kernel_launch_order(){
        return this->kernel_launch_order;
    }
    /*shoule hava some abstarct function*/

    // void launch_kernel_with_blocknum(int kernel_index, int block_complete_number) = 0;
     
};