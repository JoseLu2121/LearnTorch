#pragma once
#include "tensor.h"
#include <vector>
#include <memory>
#include <string>
#include <fstream>
#include <iostream>
#include <initializer_list>


using TensorPtr = std::shared_ptr<Tensor>; // shared pointer of a Tensor
using TensorList = std::vector<TensorPtr>; // list of shared pointers of Tensors

// Main framework class
struct Block {
    std::string name;

    Block(std::string n) : name(n) {}
    virtual ~Block() = default;

    
    virtual TensorList forward(TensorList inputs) = 0; // we force derived classes to implement forward

    
    virtual TensorList parameters() { return {}; } // by default, no parameters for blocks like Relu,Sigmoid...


    void save_weights(const std::string& filename) {

        std::ofstream out(filename, std::ios::binary);
        if(!out) {
            throw std::runtime_error("File could not been created" + filename);
        }

        auto params = this->parameters();

        size_t num_tensors = params.size();
        out.write(reinterpret_cast<const char*>(&num_tensors), sizeof(size_t));

        for(auto& tensor_ptr : params){
            tensor_ptr->serialize(out);
        }

        out.close();
        std::cout << "Weights saved in:" << filename << std::endl;

    }

    void load_weights(const std::string& filename) {

        std::ifstream in(filename, std::ios::binary);
        if(!in) {
            throw std::runtime_error("File could not been read" + filename);
        }

        auto params = this->parameters();

        size_t num_tensors_in_file;
        in.read(reinterpret_cast<char*>(&num_tensors_in_file), sizeof(size_t));

        if(num_tensors_in_file != params.size()) {
            throw std::runtime_error("Model expect" + std::to_string(params.size()) +
                "tensors, but got" + std::to_string(num_tensors_in_file));
        }

        for(auto& tensor_ptr : params) {
            tensor_ptr->deserialize(in);
        }

        in.close();
        std::cout << "Weights loaded from:" << filename << std::endl;

    }
};