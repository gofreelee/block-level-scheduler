#include "model.h"
#include <string>
#include <iostream>
#include <fstream>
model::model(std::string kernel_file_path)
{
    std::ifstream file(kernel_file_path);
    std::string kernel_source_code_content;
    int file_len;
    if (file)
    {
        file.seekg(0, file.end);
        file_len = file.tellg();
        file.seekg(0, file.beg);
        kernel_source_code_content.resize(file_len);
        file.read(&kernel_source_code_content[0], file_len);
    }
    else
    {
        std::cout << "Error opening kernel file : " << kernel_file_path << std::endl;
        exit(-1);
    }

    std::string global_str = "__global__";
    size_t pos = kernel_source_code_content.find(global_str);
    while (pos != std::string::npos)
    {
        size_t start = pos;
        pos = kernel_source_code_content.find(global_str, pos + global_str.size());
        std::string kernel_func;
        if (pos == std::string::npos)
        {
            kernel_func = kernel_source_code_content.substr(start, file_len - 1);
        }
        else
            kernel_func = kernel_source_code_content.substr(start, pos - start);
        this->kernel_source_list.push_back(kernel_func);
    }
}


