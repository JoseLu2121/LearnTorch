#include <iostream>
#include <memory>
#include <vector>
#include <cmath>
#include <string>
#include <map>
#include <set>
#include <fstream>
#include <sstream>
#include <chrono>

// Inclusiones de tu framework de Deep Learning casero
#include "tensor.h"
#include "layers.h"
#include "structure.h"
#include "loss.h"
#include "optimizer.h"
#include "trainer.h"
#include "device.h"

using namespace std;

// ============================================================================
// 1. EL TOKENIZADOR (Convierte letras a números y viceversa)
// ============================================================================
class CharTokenizer {
public:
    std::map<char, int> char_to_int;
    std::map<int, char> int_to_char;
    int vocab_size;

    CharTokenizer(const std::string& text) {
        std::set<char> unique_chars(text.begin(), text.end());
        vocab_size = 0;
        for (char c : unique_chars) {
            char_to_int[c] = vocab_size;
            int_to_char[vocab_size] = c;
            vocab_size++;
        }
    }

    std::vector<float> encode(const std::string& text) {
        std::vector<float> encoded;
        for (char c : text) {
            encoded.push_back((float)char_to_int[c]);
        }
        return encoded;
    }

    std::string decode(int token) {
        std::string decoded = "";
        decoded += int_to_char[token];
        return decoded;
    }
};

// ============================================================================
// MAIN: EL MINI-GPT (Versión Mini-Batches Anti-Cuelgues)
// ============================================================================
int main() {
    Device::set_backend(BackendType::CPU_OPTIMIZED);
    // 1. CARGAR TEXTO DESDE ARCHIVO
    cout << "Leyendo archivo 'datos.txt'..." << endl;
    ifstream archivo("datos.txt");
    
    if (!archivo.is_open()) {
        cerr << "Error: No se pudo abrir 'datos.txt'. Asegurate de crearlo en esta carpeta." << endl;
        return 1;
    }

    stringstream buffer;
    buffer << archivo.rdbuf();
    string texto_completo = buffer.str();

    cout << "Texto cargado exitosamente. Longitud: " << texto_completo.length() << " caracteres." << endl;
    CharTokenizer tokenizer(texto_completo);
    
    // Hiperparámetros
    int vocab_size = tokenizer.vocab_size;
    int max_seq_len = 128; // "Ventana" de memoria
    int embed_dim = 256;   // Capacidad del modelo

    cout << "--- Inicializando Mini-GPT en C++ ---" << endl;
    cout << "Vocabulario: " << vocab_size << " caracteres unicos." << endl;

    // 2. CREAR EL DATASET (Sliding Window)
    vector<float> x_raw;
    vector<float> y_indices;
    int num_samples = 0;
    int stride = 1;

    for (int i = 0; i <= (int)texto_completo.length() - max_seq_len - 1; i += stride) {
        string seq_x = texto_completo.substr(i, max_seq_len);
        string seq_y = texto_completo.substr(i + 1, max_seq_len);
        
        vector<float> enc_x = tokenizer.encode(seq_x);
        vector<float> enc_y = tokenizer.encode(seq_y);
        
        x_raw.insert(x_raw.end(), enc_x.begin(), enc_x.end());
        y_indices.insert(y_indices.end(), enc_y.begin(), enc_y.end());
        num_samples++;
    }

    cout << "Dataset creado: " << num_samples << " secuencias generadas." << endl;

    // 3. ENSAMBLAR LA ARQUITECTURA DEL TRANSFORMER
    auto attention_branch = make_shared<Serial>(vector<shared_ptr<Block>>{
        make_shared<SelfAttention>(embed_dim) 
    });

    auto ff_branch = make_shared<Serial>(vector<shared_ptr<Block>>{
        make_shared<Linear>(embed_dim, 4 * embed_dim),
        make_shared<ReLU>(),
        make_shared<Linear>(4 * embed_dim, embed_dim)
    });

    auto gpt = make_shared<Serial>(vector<shared_ptr<Block>>{
        make_shared<Parallel>(vector<shared_ptr<Block>>{
            make_shared<Embedding>(vocab_size, embed_dim),
            make_shared<Embedding>(max_seq_len, embed_dim)
        }),
        make_shared<Join>(JoinMode::SUM),

        make_shared<Parallel>(vector<shared_ptr<Block>>{
            make_shared<Identity>(), 
            make_shared<Serial>(vector<shared_ptr<Block>>{make_shared<LayerNorm>(embed_dim) ,attention_branch})
        }),
        make_shared<Join>(JoinMode::SUM),

        make_shared<Parallel>(vector<shared_ptr<Block>>{
            make_shared<Identity>(), 
            make_shared<Serial>(vector<shared_ptr<Block>>{make_shared<LayerNorm>(embed_dim) ,ff_branch})

        }),
        make_shared<Join>(JoinMode::SUM),

        make_shared<Linear>(embed_dim, vocab_size)
    });

    // Inicialización cuidadosa
    for (auto& p : gpt->parameters()) {
        for (int i = 0; i < p->getSize(); i++) {
            p->getData()[i] = ((float)rand()/RAND_MAX * 2.0f - 1.0f) * 0.05f;
        }
    }

    // ============================================================================
    // 4. ENTRENAMIENTO POR BATCHES (SALVAVIDAS DE LA MEMORIA RAM)
    // ============================================================================
    auto optimizer = make_shared<Adam>(gpt->parameters(), 0.001f, 0.9f, 0.999f);
    auto criterion = make_shared<CrossEntropy>();

    int batch_size = 64; // Paquetes de 64 secuencias.
    int num_batches = num_samples / batch_size;
    int epochs = 1500; // Como vemos los datos por partes, 100 épocas es buen inicio

    cout << "\nIniciando entrenamiento en " << num_batches << " mini-batches de tamaño " << batch_size << "..." << endl;
    auto inicio_entrenamiento = std::chrono::high_resolution_clock::now();
    for (int epoch = 0; epoch < epochs; epoch++) {
        float epoch_loss = 0.0f;

        for (int b = 0; b < num_batches; b++) {
            auto inicio_batch = std::chrono::high_resolution_clock::now();
            // 1. Extraer datos crudos (x)
            int start_x = b * batch_size * max_seq_len;
            int end_x = start_x + (batch_size * max_seq_len);
            vector<float> x_batch_data(x_raw.begin() + start_x, x_raw.begin() + end_x);

            // 2. CREAR Y_ONE_HOT AL VUELO
            vector<float> y_batch_data(batch_size * max_seq_len * vocab_size, 0.0f);
            for(int s = 0; s < batch_size; s++) {
                for(int t = 0; t < max_seq_len; t++) {
                    int global_s = (b * batch_size) + s;
                    int idx = (int)y_indices[global_s * max_seq_len + t];
                    y_batch_data[(s * max_seq_len * vocab_size) + (t * vocab_size) + idx] = 1.0f;
                }
            }

            // 3. Crear tensores temporales
            auto x_batch = make_shared<Tensor>(vector<int>{batch_size, max_seq_len}, x_batch_data);
            auto y_batch = make_shared<Tensor>(vector<int>{batch_size, max_seq_len, vocab_size}, y_batch_data);

            // Forward y Backward
            optimizer->zero_grad();
            auto output = gpt->forward({x_batch})[0];
            auto loss = criterion->forward(output, y_batch);
            
            loss->backward();
            optimizer->step();

            epoch_loss += loss->getData()[0];
            auto fin_batch = std::chrono::high_resolution_clock::now();
            auto tiempo_batch = std::chrono::duration_cast<std::chrono::milliseconds>(fin_batch - inicio_batch).count();

            // LA MAGIA VISUAL: '\r' hace que la línea se sobrescriba a sí misma
            cout << "\r[Epoch " << epoch << "] Batch " << b + 1 << "/" << num_batches 
                 << " | Loss: " << loss->getData()[0] 
                 << " | Tiempo: " << tiempo_batch << " ms/batch   " << flush;
            
        }

        // Imprimir progreso (cada 5 épocas para no llenar la terminal)
        if (epoch % 5 == 0 || epoch == epochs - 1) {
            cout << "Epoch " << epoch << " | Loss Media: " << epoch_loss / num_batches << endl;
        }


    }

    gpt->save_weights("basic_gpt_quijote");
    // ============================================================================
    // 5. INFERENCIA AUTOREGRESIVA (El modelo genera texto)
    // ============================================================================
    cout << "\n--- Generación de Texto ---" << endl;
    
    // Semilla inicial
    string prompt = "En un lugar de";
    vector<float> contexto(max_seq_len, 0.0f);
    auto prompt_encoded = tokenizer.encode(prompt);
    
    int pos_generacion = prompt.length();
    for(size_t i = 0; i < prompt_encoded.size(); i++) {
        contexto[i] = prompt_encoded[i];
    }
    
    cout << "Prompt: '" << prompt << "'" << endl;
    cout << "GPT escribe: " << prompt;

    int caracteres_a_generar = 60; // Que escriba un poco más esta vez

    for(int step = 0; step < caracteres_a_generar; step++) {
        auto x_gen = make_shared<Tensor>(vector<int>{1, max_seq_len}, contexto);
        auto output = gpt->forward({x_gen})[0];
        
        float max_val = -1e9;
        int token_predicho = 0;
        
        int index_a_mirar = std::min(pos_generacion - 1, max_seq_len - 1);

        float temp = 0.8f; // 1.0 = normal, 0.5 = conservador, 1.5 = locura creativa
        float max_logit = -1e9;
        
        // 1. Buscar el máximo para estabilidad numérica
        for(int v = 0; v < vocab_size; v++) {
            float val = output->getData()[index_a_mirar * vocab_size + v];
            if(val > max_logit) max_logit = val;
        }

        // 2. Calcular probabilidades con Temperatura
        vector<float> probs(vocab_size, 0.0f);
        float sum_probs = 0.0f;
        for(int v = 0; v < vocab_size; v++) {
            float val = output->getData()[index_a_mirar * vocab_size + v];
            probs[v] = std::exp((val - max_logit) / temp);
            sum_probs += probs[v];
        }

        // 3. Tirar el dado (Ruleta de probabilidades)
        float rand_val = ((float)rand() / RAND_MAX) * sum_probs;
        float acumulado = 0.0f;

        for(int v = 0; v < vocab_size; v++) {
            acumulado += probs[v];
            if(rand_val <= acumulado) {
                token_predicho = v;
                break;
            }
        }

        cout << tokenizer.decode(token_predicho) << flush;
        
        // 2. Actualizar el contexto para la siguiente letra (Sliding Window)
        if (pos_generacion < max_seq_len) {
            contexto[pos_generacion] = (float)token_predicho;
            pos_generacion++;
        } else {
            // Mover todas las letras un espacio a la izquierda
            for(int i = 0; i < max_seq_len - 1; i++) {
                contexto[i] = contexto[i+1];
            }
            // Poner la nueva letra al final
            contexto[max_seq_len - 1] = (float)token_predicho;
        }
    }
    
    cout << "\n\n¡Completado!" << endl;
    return 0;
}