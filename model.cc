#include "model.h"
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

std::vector<std::string> split(const std::string &input, char delimiter) {
    std::istringstream input_stream(input);
    std::vector<std::string> tokens;
    std::string token;

    while (std::getline(input_stream, token, delimiter)) {
        tokens.push_back(token);
    }

    return tokens;
}

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

    std::string global_str = "extern \"C\" __global__";
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

        // get kernel function from kernel_func        
        std::vector<std::string> tokens = split(kernel_func, '(');

        auto func_name_tokens = split(tokens[0], ' ');
        auto func_name = *(func_name_tokens.end() - 1);


        this->kernel_source_list[func_name] = kernel_func;
    }

    hipModuleLoad(&net_module, model_file_path);
}


