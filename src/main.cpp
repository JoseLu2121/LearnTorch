#include <iostream>
#include <memory>
#include <vector>
#include "tensor.h"
#include "layers.h"
#include "structure.h"
#include "loss.h"
#include "optimizer.h"
#include "trainer.h"
#include "ops.h"

using namespace std;

int main() {
    cout << "==========================================" << endl;
    cout << "   TEST DE CONVOLUCION + TRAINER (CNN)    " << endl;
    cout << "==========================================" << endl;

    int batch_size = 4; 

    vector<float> x_data = {

        0,0,0,0,0, 
        1,1,1,1,1, 
        0,0,0,0,0, 
        0,0,0,0,0, 
        0,0,0,0,0,

        0,0,0,0,0, 
        0,0,0,0,0, 
        0,0,0,0,0, 
        1,1,1,1,1, 
        0,0,0,0,0,

        0,1,0,0,0, 
        0,1,0,0,0, 
        0,1,0,0,0, 
        0,1,0,0,0, 
        0,1,0,0,0,

        0,0,0,1,0, 
        0,0,0,1,0, 
        0,0,0,1,0, 
        0,0,0,1,0, 
        0,0,0,1,0
    };
    

    vector<float> y_data = {
        1, 0,
        1, 0,
        0, 1,
        0, 1
    };

    auto x_train = make_shared<Tensor>(vector<int>{batch_size, 1, 5, 5}, x_data);
    auto y_train = make_shared<Tensor>(vector<int>{batch_size, 2}, y_data);

    cout << "\nConstruyendo el modelo CNN..." << endl;
    auto cnn = make_shared<Serial>(vector<shared_ptr<Block>>{
        make_shared<Conv2D>(1, 2, 3, 1, 0),
        make_shared<ReLU>(),
        make_shared<Flatten>(),
        make_shared<Linear>(18, 2)
    });

    auto optimizer = make_shared<Adam>(cnn->parameters(), 0.05f, 0.9f, 0.999f);
    auto criterion = make_shared<CrossEntropy>();

    Trainer trainer(cnn, optimizer, criterion);

    trainer.fit(x_train, y_train, 50, 1);

    cout << "\n¡Test finalizado! Si la Loss baja a 0 y el Accuracy sube a 1, tu Conv2D es perfecta." << endl;

    return 0;
}