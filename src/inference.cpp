#include <iostream>
#include <memory>
#include <vector>
#include <cmath>
#include <string>
#include <map>
#include <set>
#include <fstream>
#include <sstream>

#include "tensor.h"
#include "layers.h"
#include "structure.h"

using namespace std;


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


int main() {

    ifstream archivo("datos_dino.txt");
    if (!archivo.is_open()) {
        cerr << "Error: No se pudo abrir 'datos.txt'." << endl;
        return 1;
    }
    stringstream buffer;
    buffer << archivo.rdbuf();
    string texto_completo = buffer.str();
    CharTokenizer tokenizer(texto_completo);

    int vocab_size = tokenizer.vocab_size;
    int max_seq_len = 20;  // tarzan: 24 detective: 64
    int embed_dim = 50;    // tarzan: 32 detective: 64

    cout << "--- Cargando Mini-GPT (Modo Inferencia) ---" << endl;
    cout << "Vocabulario: " << vocab_size << " caracteres unicos." << endl;

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


    cout << "Cargando pesos desde 'basic_gpt_dino2'..." << endl;
    try {
        gpt->load_weights("basic_gpt_dino2");
        cout << "¡Pesos cargados correctamente!" << endl;
    } catch (const exception& e) {
        cerr << "Error al cargar los pesos: " << e.what() << endl;
        return 1;
    }


    string prompt = "tyran"; 
    int caracteres_a_generar = 60; 
    float temp = 0.6f; 

    cout << "\n--- Generando Texto ---" << endl;
    cout << "Temp: " << temp << " | Prompt: '" << prompt << "'" << endl;
    cout << "\nGPT escribe: " << prompt;

    vector<float> contexto(max_seq_len, 0.0f);
    auto prompt_encoded = tokenizer.encode(prompt);
    
    int pos_generacion = prompt_encoded.size();
    
    int start_idx = max(0, (int)prompt_encoded.size() - max_seq_len);
    int ctx_idx = 0;
    for(size_t i = start_idx; i < prompt_encoded.size(); i++) {
        contexto[ctx_idx++] = prompt_encoded[i];
    }

    for(int step = 0; step < caracteres_a_generar; step++) {
        auto x_gen = make_shared<Tensor>(vector<int>{1, max_seq_len}, contexto);

        auto output = gpt->forward({x_gen})[0]; 
        
        float max_val = -1e9;
        int token_predicho = 0;
        
        int index_a_mirar = std::min(pos_generacion - 1, max_seq_len - 1);

        float max_logit = -1e9;
        
        for(int v = 0; v < vocab_size; v++) {
            float val = output->getData()[index_a_mirar * vocab_size + v];
            if(val > max_logit) max_logit = val;
        }

        vector<float> probs(vocab_size, 0.0f);
        float sum_probs = 0.0f;
        for(int v = 0; v < vocab_size; v++) {
            float val = output->getData()[index_a_mirar * vocab_size + v];
            probs[v] = std::exp((val - max_logit) / temp);
            sum_probs += probs[v];
        }

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
        
        if (pos_generacion < max_seq_len) {
            contexto[pos_generacion] = (float)token_predicho;
            pos_generacion++;
        } else {
            for(int i = 0; i < max_seq_len - 1; i++) {
                contexto[i] = contexto[i+1];
            }
            contexto[max_seq_len - 1] = (float)token_predicho;
        }
    }
    
    cout << "\n\n¡Completado!" << endl;
    return 0;
}