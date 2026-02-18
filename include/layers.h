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
        W = Tensor::random({out_size, in_size},-0.01f,0.01f);
        
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
    // Constructor
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


// Sigmoid Layer
struct Sigmoid: public Block {
    // Constructor
    Sigmoid() : Block("Sigmoid") {};

    TensorList forward(TensorList inputs) override {
        if (inputs.size() != 1) {
             throw std::runtime_error("Sigmoid waits one input, it received " + std::to_string(inputs.size()));
        };
        // We return de operation sigmoid applied to the input tensor
        return { sigmoid(inputs[0]) };


    };

    // No parameters function because we have no Weights or Biases

};

// Tanh Layer
struct Tanh: public Block {
    // Constructor
    Tanh() : Block("Tanh") {};

    TensorList forward(TensorList inputs) override {
        if (inputs.size() != 1) {
             throw std::runtime_error("Sigmoid waits one input, it received " + std::to_string(inputs.size()));
        };
        // We return de operation sigmoid applied to the input tensor
        return { tanh(inputs[0]) };


    };

    // No parameters function because we have no Weights or Biases

};

// Softmax Layer
struct Softmax: public Block {
    // Constructor
    Softmax() : Block("Softmax") {};

    TensorList forward(TensorList inputs) override {
        if(inputs.size() != 1) {
            throw std::runtime_error("Softmax waits one input, it received " + std::to_string(inputs.size()));

        }
        
        auto x = inputs[0];
        
        // Estabilidad numérica: restar el máximo de cada fila
        auto axis = x->shape.size() - 1;
        auto max_x = max(x, axis);
        auto x_shifted = x - max_x; 
        
        auto exp_x = exp(x_shifted);
        auto sum_exp_x = sum(exp_x, axis);
        auto out = exp_x / sum_exp_x;

        return { out };
    }


};