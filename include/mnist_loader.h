#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <utility> 
#include "tensor.h"

using namespace std;

inline pair<shared_ptr<Tensor>, shared_ptr<Tensor>> load_mnist_csv(string filename, int max_rows = 1000) {
    ifstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("No se pudo abrir el archivo MNIST: " + filename);
    }

    string line;
    vector<float> all_pixels;
    vector<float> all_targets;
    
    // Reserva de memoria para evitar reallocations (optimización)
    all_pixels.reserve(max_rows * 784);
    all_targets.reserve(max_rows * 10);

    int rows = 0;

    // Saltamos la cabecera
    getline(file, line); 

    cout << "Cargando MNIST..." << endl;

    while (rows < max_rows && getline(file, line)) {
        stringstream ss(line);
        string val_str;
        
        // --- PARSEO SEGURO ---
        // Leemos todo a un buffer temporal primero
        vector<float> row_data;
        int label = -1;

        // 1. Label
        if (getline(ss, val_str, ',')) {
            try { label = stoi(val_str); } catch (...) { continue; }
        } else { continue; } // Línea vacía

        // 2. Píxeles
        while (getline(ss, val_str, ',')) {
            try { 
                row_data.push_back(stof(val_str) / 255.0f); 
            } catch (...) { break; }
        }

        // --- FILTRO DE INTEGRIDAD ---
        // Solo aceptamos la fila si está PERFECTA (784 píxeles)
        if (row_data.size() == 784) {
            // Copiamos al vector principal
            all_pixels.insert(all_pixels.end(), row_data.begin(), row_data.end());
            
            // Creamos target
            for (int i = 0; i < 10; i++) all_targets.push_back(i == label ? 1.0f : 0.0f);
            
            rows++;
        }
        // Si no tiene 784 (línea vacía o corrupta), la ignoramos SILENCIOSAMENTE y no subimos 'rows'
    }

    cout << "Filas válidas cargadas: " << rows << endl;

    // --- LA COMPROBACIÓN FINAL QUE EVITA EL MALLOC ERROR ---
    size_t expected_size = rows * 784;
    if (all_pixels.size() != expected_size) {
        cerr << "Error Fatal: Desajuste de memoria." << endl;
        cerr << "Esperado: " << expected_size << " | Real: " << all_pixels.size() << endl;
        throw runtime_error("Memoria corrupta en carga de datos.");
    }

    auto X = make_shared<Tensor>(vector<int>{rows, 784}, all_pixels);
    auto Y = make_shared<Tensor>(vector<int>{rows, 10}, all_targets);

    return {X, Y};
}