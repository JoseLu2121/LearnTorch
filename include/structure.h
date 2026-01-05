#pragma once
#include "block.h"
#include <vector>
#include <initializer_list>
#include <stdexcept>

// Serial class: chain of blocks
struct Serial : public Block {
    std::vector<std::shared_ptr<Block>> layers;

    // Constructor with initializer list
    Serial(std::initializer_list<std::shared_ptr<Block>> list) 
        : Block("Serial"), layers(list) {}

    // Forward: we pass the output of each layer to the input of the next
    TensorList forward(TensorList inputs) override {
        auto x = inputs;
        for (auto& layer : layers) {
            x = layer->forward(x);
        }
        return x;
    }

    // Paremeters: we concatenate the parameters of each layer
    TensorList parameters() override {
        TensorList params;
        for (auto& layer : layers) {
            auto child_params = layer->parameters();
            // .insert() concatenates vectors
            params.insert(params.end(), child_params.begin(), child_params.end());
        }
        return params;
    }
};