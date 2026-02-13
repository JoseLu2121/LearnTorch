#pragma once
#include "tensor.h"
#include <vector>

// Base Optimizer Class
struct Optimizer {
    // Parameters to optimize
    std::vector<std::shared_ptr<Tensor>> parameters;

    // Basic constructor
    Optimizer(const std::vector<std::shared_ptr<Tensor>>& params);
    virtual ~Optimizer() = default;

    // An Optimizer must have a step function to update parameters
    virtual void step() = 0; 

    // Zero out gradients for all parameters
    void zero_grad();
};

struct SGD : public Optimizer {
    float lr;

    SGD(const std::vector<std::shared_ptr<Tensor>>& params, float learning_rate);

    void step() override;
};

struct Adam : public Optimizer {
    float lr;
    float beta1;
    float beta2;
    float epsilon;
    std::vector<std::vector<float>> m_history; 
    std::vector<std::vector<float>> v_history;
    float t = 0;


    Adam(const std::vector<std::shared_ptr<Tensor>>& params, float learning_rate,
    float beta1, float beta2, float epsilon = 1e-8);
    
    void step() override;

};