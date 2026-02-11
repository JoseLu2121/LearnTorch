#include <iostream>
#include <memory>
#include <vector>
#include <cmath>
#include "tensor.h"
#include "ops.h"
#include "layers.h"
#include "loss.h"

using namespace std;

int main() {
    cout << "=== PyTorch Equivalent Test in C++ (BATCH SIZE = 2) ===" << endl << endl;
    
    // --- CONFIGURACIÓN ---
    // 1. Crear capa Linear(2, 2) con pesos manuales
    auto linear = make_shared<Linear>(2, 2);
    
    // 2. SETEAR PESOS Y BIAS A MANO
    // W = [[0.1, 0.2], [0.3, 0.4]]
    // b = [0.1, -0.1]
    auto weight = linear->W;
    weight->getData()[0] = 0.1f;  // W[0,0]
    weight->getData()[1] = 0.2f;  // W[0,1]
    weight->getData()[2] = 0.3f;  // W[1,0]
    weight->getData()[3] = 0.4f;  // W[1,1]
    
    auto bias = linear->B;
    bias->getData()[0] = 0.1f;    // b[0]
    bias->getData()[1] = -0.1f;   // b[1]
    
    cout << "Weights set to:" << endl;
    cout << "  [[0.1, 0.2]," << endl;
    cout << "   [0.3, 0.4]]" << endl;
    cout << "Bias set to: [0.1, -0.1]" << endl << endl;
    
    // --- DATOS (BATCH SIZE = 2) ---
    // Muestra 1: [1.0, 2.0] -> Target clase 0 (one-hot: [1.0, 0.0])
    // Muestra 2: [0.5, 0.5] -> Target clase 1 (one-hot: [0.0, 1.0])
    auto inputs = make_shared<Tensor>(
        vector<int>{2, 2}, 
        vector<float>{1.0f, 2.0f,    // Muestra 1
                      0.5f, 0.5f}     // Muestra 2
    );
    
    auto targets = make_shared<Tensor>(
        vector<int>{2, 2},
        vector<float>{1.0f, 0.0f,    // Target clase 0
                      0.0f, 1.0f}     // Target clase 1
    );
    
    cout << "Inputs shape: (" << inputs->shape[0] << ", " << inputs->shape[1] << ")" << endl;
    cout << "Inputs data:" << endl;
    cout << "  [[" << inputs->getData()[0] << ", " << inputs->getData()[1] << "]," << endl;
    cout << "   [" << inputs->getData()[2] << ", " << inputs->getData()[3] << "]]" << endl << endl;
    
    cout << "Targets (one-hot):" << endl;
    cout << "  [[" << targets->getData()[0] << ", " << targets->getData()[1] << "]," << endl;
    cout << "   [" << targets->getData()[2] << ", " << targets->getData()[3] << "]]" << endl << endl;
    
    // --- FORWARD ---
    // 1. Linear layer
    auto logits_list = linear->forward({inputs});
    auto logits = logits_list[0];
    
    cout << "1. Logits (Linear output):" << endl;
    cout << "   Shape: (" << logits->shape[0] << ", " << logits->shape[1] << ")" << endl;
    cout << "   Data:" << endl;
    cout << "   [[" << logits->getData()[0] << ", " << logits->getData()[1] << "]," << endl;
    cout << "    [" << logits->getData()[2] << ", " << logits->getData()[3] << "]]" << endl;
    cout << "   Expected:" << endl;
    cout << "   [[0.6, 1.0],    # 0.1*1 + 0.2*2 + 0.1 = 0.6, 0.3*1 + 0.4*2 - 0.1 = 1.0" << endl;
    cout << "    [0.25, 0.25]]  # 0.1*0.5 + 0.2*0.5 + 0.1 = 0.25, 0.3*0.5 + 0.4*0.5 - 0.1 = 0.25" << endl;
    cout << string(70, '-') << endl << endl;
    
    // 2. Softmax
    auto softmax = make_shared<Softmax>();
    auto probs_list = softmax->forward({logits});
    auto probs = probs_list[0];
    
    cout << "2. Probabilities (Softmax):" << endl;
    cout << "   Shape: (" << probs->shape[0] << ", " << probs->shape[1] << ")" << endl;
    cout << "   Data:" << endl;
    cout << "   [[" << probs->getData()[0] << ", " << probs->getData()[1] << "]," << endl;
    cout << "    [" << probs->getData()[2] << ", " << probs->getData()[3] << "]]" << endl;
    cout << string(70, '-') << endl << endl;
    
    // 3. CrossEntropy Loss (con promedio sobre el batch)
    auto criterion = make_shared<CrossEntropy>();
    auto loss = criterion->forward(probs, targets);
    
    cout << "3. Loss (CrossEntropy with mean reduction):" << endl;
    cout << "   Value: " << loss->getData()[0] << endl;
    cout << "   This is the AVERAGE loss over the 2 samples" << endl;
    cout << string(70, '-') << endl << endl;
    
    // --- BACKWARD ---
    cout << "4. Computing Gradients (Backward Pass)..." << endl;
    
    // Limpiar gradientes previos (simular zero_grad)
    if (weight->grad) {
        for (size_t i = 0; i < weight->getSize(); i++) {
            weight->grad->getData()[i] = 0.0f;
        }
    }
    if (bias->grad) {
        for (size_t i = 0; i < bias->getSize(); i++) {
            bias->grad->getData()[i] = 0.0f;
        }
    }
    
    // Ejecutar backward
    loss->backward();
    
    cout << "   Done!" << endl << endl;
    
    // --- RESULTADOS ---
    cout << "5. Gradients of Weights (dL/dW):" << endl;
    if (weight->grad) {
        cout << "   [[" << weight->grad->getData()[0] << ", " << weight->grad->getData()[1] << "]," << endl;
        cout << "    [" << weight->grad->getData()[2] << ", " << weight->grad->getData()[3] << "]]" << endl;
    } else {
        cout << "   No gradient computed!" << endl;
    }
    cout << endl;
    
    cout << "6. Gradients of Bias (dL/db):" << endl;
    if (bias->grad) {
        cout << "   [" << bias->grad->getData()[0] << ", " << bias->grad->getData()[1] << "]" << endl;
    } else {
        cout << "   No gradient computed!" << endl;
    }
    
    cout << "\n" << string(70, '=') << endl;
    cout << "=== Test Complete ===" << endl;
    
    return 0;
}