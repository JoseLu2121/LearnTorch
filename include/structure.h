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

    // Constructor with vector ( friendly)
    Serial(const std::vector<std::shared_ptr<Block>>& list)
        : Block("Serial"), layers(list) {}

    // Forward: we pass the output of each layer to the input of the next
    TensorList forward(TensorList inputs) override {
        TensorList outputs;
        auto x = inputs;
        for(auto& layer : layers) {
            auto out_layer = layer->forward(x);
            outputs.insert(outputs.end(),out_layer.begin(), out_layer.end());
        }
        return outputs;
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

struct Parallel : public Block {
    std::vector<std::shared_ptr<Block>> layers;
    // Constructor with initializer list
    Parallel(std::initializer_list<std::shared_ptr<Block>> list) 
        : Block("Parallel"), layers(list) {}

    // Constructor with vector ( friendly)
    Parallel(const std::vector<std::shared_ptr<Block>>& list)
        : Block("Parallel"), layers(list) {}

    TensorList forward(TensorList inputs) override {
        TensorList output;
        for(auto& layer : layers) {
            output.push_back(layer->forward(inputs));
        }
        return output;
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


struct Join : public Block {
    Join(std::initializer_list<std::shared_ptr<Block>> list, JoinMode m = JoinMode::SUM) 
        : Block("Join"), layers(list) {}

    // Constructor with vector ( friendly)
    Join(const std::vector<std::shared_ptr<Block>>& list, JoinMode m = JoinMode::SUM)
        : Block("Join"), layers(list) {}

    TensorList forward(TensorList inputs) override {
        if (inputs.empty()) return {};

        if(mode == JoinMode::SUM) {
            auto accum = inputs[0];
            for(size_t i = 1; i < inputs.size() ; i++){
                accum = accum + accum[i];

            }

            return { accum };
        }

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