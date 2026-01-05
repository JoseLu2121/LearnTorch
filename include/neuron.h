#pragma once
#include "tensor.h"

// function that has a tensor as input and returns a tensor
using Activation = std::function<std::shared_ptr<Tensor>(std::shared_ptr<Tensor>)>;

struct Neuron : std::enable_shared_from_this<Neuron> {

    public:
        int input_size;

        std::shared_ptr<Tensor> weights;
        std::shared_ptr<Tensor> bias;
        std::shared_ptr<Tensor> output;
        Activation activation_function;
    
    Neuron(int input_size, Activation activation_function = nullptr);
    
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input);
};