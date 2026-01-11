#include "trainer.h"
#include <iostream>

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

        // 4. Backward
        loss->backward();

        // 5. Update
        optimizer->step();

        // 6. Logging
        if (epoch % print_every == 0 || epoch == epochs - 1) {
            std::cout << "Epoch " << epoch << " | Loss: " << loss->getData()[0] << std::endl;
        }
    }
    
    std::cout << "--- Training finalised---" << std::endl;
}