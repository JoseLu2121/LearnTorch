#pragma once
#include "tensor.h"
#include "block.h"
#include "optimizer.h"
#include "loss.h"
#include <memory>

class Trainer {
private:
    std::shared_ptr<Block> model;
    std::shared_ptr<Optimizer> optimizer;
    std::shared_ptr<Loss> criterion;

public:
    
    Trainer(std::shared_ptr<Block> m, 
            std::shared_ptr<Optimizer> o, 
            std::shared_ptr<Loss> l);

    void fit(std::shared_ptr<Tensor> x_train, 
             std::shared_ptr<Tensor> y_train, 
             int epochs, 
             int print_every = 10);
};