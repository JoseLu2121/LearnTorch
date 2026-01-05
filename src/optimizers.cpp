#include "optimizer.h"
#include <algorithm>


Optimizer::Optimizer(const std::vector<std::shared_ptr<Tensor>>& params)
    : parameters(params) {}


// Zero Grad
void Optimizer::zero_grad() {
    for(auto& param : parameters){
        if(param->grad){
            std::fill(param->grad->getData(), 
                      param->grad->getData() + param->grad->getSize(), 
                      0.0f);
        }
    }
}


SGD::SGD(const std::vector<std::shared_ptr<Tensor>>& params, float learning_rate)
    : Optimizer(params), lr(learning_rate) {}

// Step function: update parameters using gradients
void SGD::step() {
    for (auto& param : parameters) {
        if (param->grad) {
            float* param_data = param->getData();
            float* grad_data = param->grad->getData();
            size_t size = param->getSize();

            for (size_t i = 0; i < size; ++i) {
                // we add to the parameter the negative gradient scaled by learning rate
                param_data[i] -= lr * grad_data[i]; 
            }
        }
    }
}






