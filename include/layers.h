#pragma once
#include "block.h"
#include "ops.h"       
#include <stdexcept> 

// Linear Layer
// Y = X @ W.T + B
struct Linear : public Block {
    TensorPtr W; // Weight (Out, In)
    TensorPtr B; // Bias (1, Out)

    // Constructor
    Linear(int in_size, int out_size) : Block("Linear") {
        // Shape: (Out,In)
        W = Tensor::random({out_size, in_size});
        
        // Shape: (1,Out)
        B = Tensor::zeros({1, out_size});
    }

    // Forward function
    TensorList forward(TensorList inputs) override {
        // We expect exactly one input tensor
        if (inputs.size() != 1) {
            throw std::runtime_error("Linear layer expects one input, it received " + std::to_string(inputs.size()));
        }

        auto X = inputs[0];

        // we use the transpose operation to get W.T
        auto W_t = transpose_view(W);

        // Compute Y = X @ W.T + B
        auto Y = matmul(X, W_t) + B;

        return {Y}; // We convert it to TensorList
    }

    // Parameters function (we add Weights and Bias)
    std::vector<TensorPtr> parameters() override {
        return {W, B};
    }
};

// Relu Layer
// Y = max(0, X)
struct ReLU : public Block {
    // Constructor (no Weights or Biases)
    ReLU() : Block("ReLU") {}

    TensorList forward(TensorList inputs) override {
        if (inputs.size() != 1) {
             throw std::runtime_error("ReLU waits one input, it received " + std::to_string(inputs.size()));
        }
        
        // We return de operation relu applied to the input tensor
        return { relu(inputs[0]) };
    }

    // No parameters function because we have no Weights or Biases
};