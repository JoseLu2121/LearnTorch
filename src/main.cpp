#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include "tensor.h"
#include "block.h"
#include "layers.h"
#include "structure.h"
#include "optimizer.h"
#include "loss.h"
#include "trainer.h"
#include "mnist_loader.h"

using namespace std;

// --- FUNCIÓN PARA VER LAS IMÁGENES ---
void visualize_results(shared_ptr<Block> model, shared_ptr<Tensor> x, shared_ptr<Tensor> y, int num_samples) {
    cout << "\n=== VISUALIZACION DE RESULTADOS ===" << endl;
    
    auto preds = model->forward({x})[0];
    float* p_data = preds->getData();
    float* x_data = x->getData();
    float* y_data = y->getData();
    
    int total_samples = x->shape[0];

    for(int k=0; k<num_samples; k++) {
        // Elegir un índice aleatorio o los primeros k
        int idx = k; // O rand() % total_samples;
        
        // Obtener predicción y real
        float* this_pred = p_data + idx * 10;
        float* this_true = y_data + idx * 10;
        
        // Argmax casero
        int p_digit = 0; float max_p = -999;
        int t_digit = 0; float max_t = -999;
        
        for(int i=0; i<10; i++) {
            if(this_pred[i] > max_p) { max_p = this_pred[i]; p_digit = i; }
            if(this_true[i] > max_t) { max_t = this_true[i]; t_digit = i; }
        }

        cout << "\nEjemplo #" << idx << " | Real: " << t_digit << " | Prediccion: " << p_digit;
        cout << (p_digit == t_digit ? " [CORRECTO]" : " [FALLO]") << endl;
        cout << "----------------------------" << endl;

        // DIBUJAR DIGITO (ASCII ART)
        float* pixels = x_data + idx * 784;
        for(int r=0; r<28; r++) {
            for(int c=0; c<28; c++) {
                float val = pixels[r*28 + c];
                if (val > 0.75) cout << "@";      // Blanco fuerte
                else if (val > 0.5) cout << "%";  // Gris
                else if (val > 0.2) cout << ".";  // Gris claro
                else cout << " ";                 // Negro
                cout << " "; // Espaciado para que no se vea aplastado
            }
            cout << endl;
        }
        cout << "----------------------------" << endl;
    }
}

int main() {
    // 1. CARGAR DATOS
    // Cargamos 1000 ejemplos para entrenar rápido
    auto dataset = load_mnist_csv("../test/mnist_train.csv", 1000); 
    auto x_train = dataset.first;
    auto y_train = dataset.second;

    cout << "Datos cargados: " << x_train->shape[0] << " imagenes." << endl;

    // 2. MODELO
    auto model = make_shared<Serial>(initializer_list<shared_ptr<Block>>{
        make_shared<Linear>(784, 64),
        make_shared<ReLU>(),
        make_shared<Linear>(64, 10)
    });

    // --- INICIALIZACIÓN DE PESOS (IMPORTANTE) ---
    srand(time(0));
    for(auto& p : model->parameters()) {
        float* d = p->getData();
        // Inicialización Xavier simplificada
        float scale = sqrt(2.0f / (p->getDimension() > 1 ? p->shape[1] : 1));
        if (p->getDimension() == 1) scale = 0.0f; // Bias a 0
        
        for(int i=0; i<p->getSize(); i++) {
            d[i] = ((float)rand()/RAND_MAX - 0.5f) * 2 * scale * 0.1f; 
        }
    }
    // --------------------------------------------

    // 3. ENTRENAR
    // Usamos LR = 0.1 porque MSE con OneHot a veces requiere empujón fuerte
    auto opt = make_shared<SGD>(model->parameters(), 0.1f); 
    auto loss_fn = make_shared<MSELoss>();

    Trainer trainer(model, opt, loss_fn);
    
    // 50 Epochs para asegurar que baje bien
    trainer.fit(x_train, y_train, 200, 5); 

    // 4. VER QUE HA PASADO
    visualize_results(model, x_train, y_train, 3); // Visualizar 3 ejemplos

    return 0;
}