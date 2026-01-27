#pragma once
#include "tensor.h"
#include <memory>

// Base loss functions class
struct Loss {
    virtual std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> predictions, std::shared_ptr<Tensor> targets) = 0;
    virtual ~Loss() = default;
};


struct MSELoss : public Loss {
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> prediction, std::shared_ptr<Tensor> target) override;
};

struct CrossEntropy : public Loss {
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> prediction, std::shared_ptr<Tensor> target) override;
};
