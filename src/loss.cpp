#include "ops.h"
#include "loss.h"
#include "tensor.h"


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

        auto func =  (log(prediction + 1e-8f) *  target) * -1.0f;
        auto loss_per_sample = sum(func, 1);
        auto total_sum = sum(loss_per_sample, 0);
        
        float batch_size = (float)prediction->shape[0];
        
        return total_sum / batch_size;
    }