#pragma once
#include "tensor.h"
#include <vector>
#include <memory>
#include <string>
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
};