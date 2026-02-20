#include <iostream>
#include <memory>
#include <vector>
#include <cmath>
#include "tensor.h"
#include "ops.h"
#include "layers.h"
#include "loss.h"
#include "structure.h"
#include "trainer.h"
#include "optimizer.h"

using namespace std;

int main() {    
    // --- CONFIGURACIÓN ---
    auto linear = make_shared<Linear>(2, 4);
    auto relu = make_shared<ReLU>();
    auto linear2 = make_shared<Linear>(4, 2);
    auto softmax = make_shared<Softmax>();
    auto model = std::make_shared<Serial>(std::vector<std::shared_ptr<Block>>{linear, relu, linear2, softmax});
    auto adam = make_shared<Adam>(model->parameters(),0.01f,0.9f,0.999f,1e-8f);
    auto sgd = make_shared<SGD>(model->parameters(),0.01f);
    auto criterion = make_shared<CrossEntropy>();
    auto trainer = new Trainer(model,adam, criterion);

    auto inputs = make_shared<Tensor>(
        vector<int>{2, 3, 2}, 
        vector<float>{
            // Batch 0
            1.0f, 2.0f,   // Seq 0
            0.5f, 0.5f,   // Seq 1
           -1.0f, 0.0f,   // Seq 2
            // Batch 1
            0.0f,-1.0f,   // Seq 0
            2.0f, 1.0f,   // Seq 1
            1.0f, 1.0f    // Seq 2
        }
    );


    auto targets = make_shared<Tensor>(
        vector<int>{2, 3, 2},
        vector<float>{
            // Batch 0 (One-hot)
            1.0f, 0.0f,   // Target clase 0
            0.0f, 1.0f,   // Target clase 1
            1.0f, 0.0f,   // Target clase 0
            // Batch 1 (One-hot)
            0.0f, 1.0f,   // Target clase 1
            1.0f, 0.0f,   // Target clase 0
            0.0f, 1.0f    // Target clase 1
        }
    );

    trainer->fit(inputs,targets,200);

    model->save_weights("model_weights");


    auto linear3 = make_shared<Linear>(2, 4);
    auto relu2 = make_shared<ReLU>();
    auto linear4 = make_shared<Linear>(4, 2);
    auto softmax2 = make_shared<Softmax>();
    auto model2 = std::make_shared<Serial>(std::vector<std::shared_ptr<Block>>{linear, relu, linear2, softmax});
    auto adam2 = make_shared<Adam>(model->parameters(),0.01f,0.9f,0.999f,1e-8f);
    auto sgd2 = make_shared<SGD>(model->parameters(),0.01f);
    
    model2->load_weights("model_weights");

    TensorList outputs = model2->forward({inputs});
    auto prediction = outputs[0];
    prediction->printElements(12);
    return 0;
}