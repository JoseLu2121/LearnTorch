#include "ops.h"
#include "loss.h"
#include "tensor.h"
#include <iostream>

using namespace std;

std::shared_ptr<Tensor> MSELoss::forward(std::shared_ptr<Tensor> prediction, std::shared_ptr<Tensor> target) {
    
    auto diff = prediction - target;
    auto sq_error = diff * diff;
    auto sum_features = sum(sq_error, 1);
    auto total_error = sum(sum_features, 0);
    
    float N = (float)prediction->getSize();
    
    return total_error / N;
}

std::shared_ptr<Tensor> CrossEntropy::forward(std::shared_ptr<Tensor> prediction, 
    std::shared_ptr<Tensor> target) {

        float total_samples = (float)(prediction->getSize() / prediction->shape.back());
        float sum_val = 0.0f;
        for(size_t i = 0; i < prediction->getSize(); i++) {
            float p = prediction->getData()[i] + 1e-8f;
            float y = target->getData()[i];
            sum_val += -y * log(p);
        }
        float final_loss = sum_val / total_samples;
        auto out = std::make_shared<Tensor>(std::vector<int>{1}, std::vector<float>{final_loss});
        out->parents = {prediction};

        out->_backward = [prediction,target,total_samples]() {

            if(!prediction->grad) {
                prediction->grad = std::make_shared<Tensor>(prediction->shape, std::vector<float>(prediction->getSize(),0.0f));
            }

            for(size_t i = 0; i < prediction->getSize(); i++) {
                float p = prediction->getData()[i] + 1e-8f;
                float y = target->getData()[i];

                prediction->grad->getData()[i] += (-y/p) / total_samples;
            };


        };

        return out;
    }