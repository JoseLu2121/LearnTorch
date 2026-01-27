#include "backend.h"
#include "types.h"
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>
#include <string>

using namespace std;

size_t element_vector_product(const std::vector<int>& v) {
    return std::accumulate(v.begin(), v.end(), 1, std::multiplies<int>());
}

// Get memory index given one
inline int getIndex(int index, const TensorInfo& tensor){
    int offset = 0;
    int current_index = index;
    // We iterate each dimension from the right
    // FIX: d >= 0 para incluir la dimensión 0
    for(int d = tensor.dim - 1; d >= 0 ; --d){
        int relative_position = current_index % tensor.shape[d]; // Get the relative position from the actual dim
        offset += relative_position * tensor.strides[d]; // Apply stride in case of broadcast
        current_index /= tensor.shape[d]; // We establish the current index of the next dimension
    }

    return offset;

}


// A basic binary operation of two tensors
void CPUBackend::binary(const TensorInfo& a, const TensorInfo& b, TensorInfo& out, BinaryOp op){

    for(size_t i = 0; i < out.size; i++){ // size_t para evitar warnings
        // Get the real index of a,b and out
        int a_index = getIndex(i,a);
        int b_index = getIndex(i,b);
        int out_index = getIndex(i,out);

        // Get the values
        float out_val = 0.0f;
        float a_val = a.data[a_index];
        float b_val = b.data[b_index];

        // Switch for each case of operation
        switch (op)
        {
        case BinaryOp::ADD : out_val = a_val + b_val; break;
        case BinaryOp::MUL : out_val = a_val * b_val; break;
        case BinaryOp::SUB : out_val = a_val - b_val; break;
        case BinaryOp::DIV : out_val = a_val / b_val; break;
        case BinaryOp::POW : out_val = std::pow(a_val,b_val); break;
        }
        
        out.data[out_index] = out_val;
    };

};

// A basic unary operation of a tensor
void CPUBackend::unary(const TensorInfo& a, TensorInfo& out, UnaryOp op){

    for(int i = 0; i < out.size; i++){

        int a_index = getIndex(i,a);
        int out_index = getIndex(i,out);
        
        float a_val = a.data[a_index];
        float out_val = 0.0f;
        
        switch (op)
        {
            case UnaryOp::RELU : out_val = (a_val > 0.0f) ? a_val : 0.0f; break;
            case UnaryOp::SIGMOID : out_val = 1.0f / (1.0f + std::exp(-a_val)); break;
            case UnaryOp::TANH: out_val = std::tanh(a_val); break;
            case UnaryOp::EXP : out_val = std::exp(a_val); break;
            case UnaryOp::LOG : out_val = std::log(a_val); break;
            case UnaryOp::NEG : out_val = -a_val; break;
        }
        
        out.data[out_index] = out_val;
    }
}



void CPUBackend::gemm(const TensorInfo& a, const TensorInfo& b, TensorInfo& out) {
    auto n_batch = out.shape[0];
    auto M = out.shape[1];
    auto N = out.shape[2];
    auto K = a.shape[2];

    // We iterate per batch 
    for(int b_idx = 0; b_idx < n_batch; b_idx++){
        const float* batch_a = a.data + b_idx * a.strides[0];
        const float* batch_b = b.data + b_idx * b.strides[0];
        float* batch_out = out.data + b_idx * out.strides[0];

        for(int m = 0; m < M; m++){
            for(int n = 0; n < N; n++){
                float sum = 0.0f;
                const float* row_a = batch_a + m * a.strides[1];
                const float* col_b = batch_b + n * b.strides[2];

                for(int k = 0; k<K; k++){
                    float val_a = row_a[k * a.strides[2]];
                    float val_b = col_b[k * b.strides[1]];
                    sum += val_a * val_b;
                }
                batch_out[m * out.strides[1] + n * out.strides[2]] = sum;
            }

        }
    }
}

// === Implementaciones Dummy para satisfacer al Linker ===
float* CPUBackend::alloc(size_t size) { return new float[size]; }
void CPUBackend::free(float* ptr) { delete[] ptr; }
void CPUBackend::set(float* ptr, float value, size_t size) { /* TODO */ }
void CPUBackend::reduce(const TensorInfo& a, const TensorInfo& b, ReduceOp op) { /* TODO */ }


// Unbroadcast function to accumulate gradients correctly
void CPUBackend::accumulate_grad(shared_ptr<Tensor> param, shared_ptr<Tensor> incoming_grad) {

    float* p_data = param->grad->getData(); // Buffer to accumulate gradients
    float* g_data = incoming_grad->getData(); // Incoming gradient data that needs to be unbroadcasted

    const auto& p_shape = param->shape;
    const auto& p_strides = param->strides;
    const auto& g_shape = incoming_grad->shape;
   
    int g_dims = g_shape.size();
    int p_dims = p_shape.size();
    int dim_diff = g_dims - p_dims; 

    // Coordinates of the incoming gradient
    vector<int> g_coords(g_dims, 0); // ex: if g_dims = 3, g_coords = {0,0,0}

    size_t total = incoming_grad->getSize();

    for (size_t idx = 0; idx < total; idx++) {

        // we calculate the offset
        int p_offset = 0;
        for (int j = 0; j < p_dims; j++) { // we iterate each dimension of the original gradient
            int g_j = j + dim_diff;        // we look up the corresponding dimension in the incoming gradient
            // if the parameter shape in this dimension is 1, we always take coordinate 0 (broadcasted)
            int coord = (p_shape[j] == 1) ? 0 : g_coords[g_j];
            p_offset += coord * p_strides[j];
        }

        // we accumulate the gradient
        p_data[p_offset] += g_data[idx];

        // We increment like an odometer ex for 3D: (0,0,0) -> (0,0,1)
        for (int d = g_dims - 1; d >= 0; d--) {
            g_coords[d]++;
            if (g_coords[d] < g_shape[d]) break;
            g_coords[d] = 0;
        }
    }
}
