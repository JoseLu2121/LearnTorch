#include "ops.h"
#include "loss.h"
#include "tensor.h"


std::shared_ptr<Tensor> MSELoss::forward(std::shared_ptr<Tensor> prediction, std::shared_ptr<Tensor> target) {
    
    auto diff = prediction - target;
    auto sq_error = diff * diff;
    

    auto total_error = sum(sq_error); 
    
    float N = (float)sq_error->getSize();
    
    return total_error * (1.0f / N);
}