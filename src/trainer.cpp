#include "trainer.h"
#include <iostream>
#include "ops.h"

// Trainer Constructor
Trainer::Trainer(std::shared_ptr<Block> m, 
                 std::shared_ptr<Optimizer> o, 
                 std::shared_ptr<Loss> l) 
    : model(m), optimizer(o), criterion(l) {}


// Fit Method: Train the model
void Trainer::fit(std::shared_ptr<Tensor> x_train, 
                  std::shared_ptr<Tensor> y_train, 
                  int epochs, 
                  int print_every) {

    std::cout << "--- Starting training (" << epochs << " epochs) ---" << std::endl;

    for (int epoch = 0; epoch < epochs; epoch++) {
        
        // 1. Zero Grad everything
        optimizer->zero_grad();

        // 2. Forward
        auto outputs = model->forward({x_train});
        auto prediction = outputs[0];

        // 3. Loss
        auto loss = criterion->forward(prediction, y_train);

        auto accuracy = calculate_accuracy(prediction,y_train);

        // 4. Backward
        loss->backward();

        // 5. Update
        optimizer->step();

        // 6. Logging
        if (epoch % print_every == 0 || epoch == epochs - 1) {
            std::cout << "Epoch " << epoch << " | Loss: " << loss->getData()[0] << " | Accuracy:" <<  accuracy << std::endl;
        }
    }
    
    std::cout << "--- Training finalised---" << std::endl;
}

float  Trainer::calculate_accuracy(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    std::shared_ptr<Tensor> a_predictions;
    std::shared_ptr<Tensor> b_labels;
    if(a->getDimension() == 2){
        a_predictions = argmax(a,1);

    } else {
        a_predictions = argmax(a,2);
    }

    if(b->getDimension() == a->getDimension()){
        if(b->getDimension() == 2){
            b_labels = argmax(b, 1);
        } else {
            b_labels = argmax(b, 2);
        }
    } else {
        b_labels = b;
    }
    int count_success = 0;
    
    for(size_t i = 0; i < a_predictions->getSize(); i++){
        if(a_predictions->getData()[i] == b_labels->getData()[i]){
            count_success++;
        }

    }

    auto accuracy = (float) count_success / a_predictions->getSize();

    return accuracy;

}