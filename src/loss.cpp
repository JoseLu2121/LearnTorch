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
    
    // 1. Calculamos Softmax por cada fila (token) para estabilidad
    // Probabilidades = exp(logits) / sum(exp(logits))
    int batch_seq = prediction->getSize() / prediction->shape.back();
    int feat_size = prediction->shape.back();
    std::vector<float> probs(prediction->getSize());

    for (int b = 0; b < batch_seq; b++) {
        float max_val = -1e9;
        for (int f = 0; f < feat_size; f++) 
            max_val = std::max(max_val, prediction->getData()[b * feat_size + f]);

        float sum_exp = 0;
        for (int f = 0; f < feat_size; f++) {
            probs[b * feat_size + f] = std::exp(prediction->getData()[b * feat_size + f] - max_val);
            sum_exp += probs[b * feat_size + f];
        }
        for (int f = 0; f < feat_size; f++) 
            probs[b * feat_size + f] /= (sum_exp + 1e-9f);
    }

    // 2. Loss: -sum(target * log(probs))
    float sum_val = 0.0f;
    for(size_t i = 0; i < prediction->getSize(); i++) {
        if (target->getData()[i] > 0.0f) {
            sum_val += -target->getData()[i] * std::log(std::max(probs[i], 1e-12f));
        }
    }

    float final_loss = sum_val / batch_seq;
    auto out = std::make_shared<Tensor>(std::vector<int>{1}, std::vector<float>{final_loss});
    out->parents = {prediction};

    // 3. BACKWARD: La magia (Predicción - Target)
    out->_backward = [prediction, target, probs, batch_seq]() {
        if(!prediction->grad) {
            prediction->grad = std::make_shared<Tensor>(prediction->shape, std::vector<float>(prediction->getSize(), 0.0f));
        }
        for(size_t i = 0; i < prediction->getSize(); i++) {
            // El gradiente de Softmax+CE es simplemente (P - Y)
            float grad_val = (probs[i] - target->getData()[i]) / batch_seq;
            prediction->grad->getData()[i] += grad_val;
        }
    };

    return out;
}