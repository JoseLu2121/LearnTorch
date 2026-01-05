#include <iostream>
#include <vector>
#include <memory>
#include "tensor.h"
#include "block.h"
#include "layers.h"
#include "structure.h"
#include "optimizer.h"
#include "loss.h"
#include "trainer.h"

using namespace std;

// Helper para forzar pesos
void set_data(shared_ptr<Tensor> t, vector<float> vals) {
    float* ptr = t->getData();
    for(size_t i=0; i<vals.size(); i++) ptr[i] = vals[i];
}

int main() {
    cout << "--- C++ FRAMEWORK TRAINING ---" << endl;

    // 1. DATOS (Batch size = 2)
    // Input: [[1.0], [2.0]] (Shape 2x1)
    auto x_train = Tensor::zeros({2, 1});
    set_data(x_train, {1.0f, 2.0f});

    // Target: [[2.0], [4.0]] (Shape 2x1)
    auto y_train = Tensor::zeros({2, 1});
    set_data(y_train, {2.0f, 4.0f});

    // 2. MODELO
    // Linear(1->2) -> ReLU -> Linear(2->1)
    auto l1 = make_shared<Linear>(1, 2);
    auto relu = make_shared<ReLU>();
    auto l2 = make_shared<Linear>(2, 1);

    auto model = make_shared<Serial>(initializer_list<shared_ptr<Block>>{l1, relu, l2});

    // 3. FORZAR PESOS INICIALES (Determinismo)
    // Capa 1: W (2x1), B (1x2)
    set_data(l1->W, {0.5f, -0.5f});
    set_data(l1->B, {0.1f, -0.1f});

    // Capa 2: W (1x2), B (1x1)
    set_data(l2->W, {1.0f, -1.0f});
    set_data(l2->B, {0.0f});

    // 4. CONFIGURACIÓN
    // SGD con Learning Rate 0.05
    auto opt = make_shared<SGD>(model->parameters(), 0.05f);
    auto loss_fn = make_shared<MSELoss>();

    // 5. ENTRENAMIENTO
    Trainer trainer(model, opt, loss_fn);
    
    // Entrenamos 10 epochs e imprimimos TODAS (print_every=1)
    trainer.fit(x_train, y_train, 10, 1);

    // Predicción final
    cout << "\nPrediccion final:" << endl;
    auto pred = model->forward({x_train})[0];
    pred->printElements(2);

    return 0;
}