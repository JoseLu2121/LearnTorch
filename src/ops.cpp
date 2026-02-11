#include "ops.h"
#include "types.h"
#include "utils.h"
#include "backend.h"
#include "device.h"
#include <algorithm>
#include <stdexcept>
#include <cmath> 
#include <iostream> // Necesario para cout

using namespace std;


std::shared_ptr<Tensor> linear_activation(std::shared_ptr<Tensor> x) {
    return x; // Identity function
}

shared_ptr<Tensor> operator+(shared_ptr<Tensor> a, shared_ptr<Tensor> b) {
    auto out = a->compute_binary_op(b, BinaryOp::ADD);

    out->parents = {a,b};

    out->_backward = [a, b, out]() {
        if (!a->grad) a->grad = Tensor::zeros(a->shape);
        if (!b->grad) b->grad = Tensor::zeros(b->shape);
        Device::get()->accumulate_grad(a, out->grad);
        Device::get()->accumulate_grad(b, out->grad);
    };

    return out;
}

shared_ptr<Tensor> matmul(shared_ptr<Tensor> a, shared_ptr<Tensor> b, bool require_grad) {
    // Tranform the tensor to an optimized view of it for gemm
    auto a_view = a->view_to_gemm(false);
    auto b_view = b->view_to_gemm(true);
    // The out batch will be the max of a and b
    int batch_out = std::max(a_view->shape[0], b_view->shape[0]);
    
    // 1. Determine the REAL desired output shape (2D vs 3D)
    std::vector<int> out_shape;
    if (a->shape.size() == 2 && b->shape.size() == 2) {
        // Linear algebra case: (M, K) @ (K, N) -> (M, N)
        out_shape = {a_view->shape[1], b_view->shape[2]};
    } else {
        // Batch case
        out_shape = {batch_out, a_view->shape[1], b_view->shape[2]};
    }

    // 2. Create output tensor with the CORRECT shape initially
    auto out = std::make_shared<Tensor>(out_shape);
    out->parents = {a, b}; 

    // 3. Create a temporary GEMM view of the output for the backend
    // Since 'out' data is empty, we just need a view that fits the GEMM interface (Batch, M, N)
    shared_ptr<Tensor> out_view;
    if (out_shape.size() == 2) {
       out_view = out->view_to_gemm(false); 
    } else {
       out_view = out; 
    }

    TensorInfo i_a = a_view.get()->getInfo();
    TensorInfo i_b = b_view.get()->getInfo();
    TensorInfo i_out = out_view.get()->getInfo();
    
    // Backend gemm call
    static CPUBackend backend;
    backend.gemm(i_a,i_b, i_out);

    out->_backward = [a, b, out]() {
        if (!a->grad) a->grad = Tensor::zeros(a->shape);
        if (!b->grad) b->grad = Tensor::zeros(b->shape);
        auto grad_output = out->grad;
        
        // dA = Grad @ B.T
        // Matriz (Batch, Out) @ (Out, In) -> (Batch, In)
        auto b_T = transpose_view(b);
        auto da = matmul(grad_output, b_T,false);
        Device::get()->accumulate_grad(a, da);

        // dB = A.T @ Grad
        // Matrix (In, Batch) @ (Batch, Out) -> (In, Out)
        auto a_T = transpose_view(a);
        auto db = matmul(a_T, grad_output, false);
        
        // db is (In, Out). b is (In, Out).
        // No transposition needed.
        
        Device::get()->accumulate_grad(b, db);
    };

    return out;
}


std::shared_ptr<Tensor> operator*(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) { 
    
    auto out = a->compute_binary_op(b, BinaryOp::MUL);

    out->parents = {a,b};

    out->_backward = [a, b, out]() { 
        if (!a->grad) a->grad = Tensor::zeros(a->shape);
        if (!b->grad) b->grad = Tensor::zeros(b->shape); 

        auto grad_a_part = out->grad * b; 
        Device::get()->accumulate_grad(a, grad_a_part);

        auto grad_b_part = out->grad * a;
        Device::get()->accumulate_grad(b, grad_b_part);

        };

    return out;
    
}



shared_ptr<Tensor> operator*(shared_ptr<Tensor> a, float scalar) {
    // Forward
    vector<float> output_data(element_vector_product(a->getShape()));
    float* input_data = a->getData();
    size_t size = a->getSize();

    for (size_t i = 0; i < size; i++) {
        output_data[i] = input_data[i] * scalar;
    }

    auto result = make_shared<Tensor>(a->getShape(), output_data , vector<shared_ptr<Tensor>>{a});

    // Backward: dy/dx = scalar
    result->_backward = [a, result, scalar]() {
        if (!a->grad) a->grad = Tensor::zeros(a->shape);

        float* grad_input = a->grad->getData();
        float* grad_output = result->grad->getData();
        size_t size = a->getSize();

        for (size_t i = 0; i < size; i++) {
            grad_input[i] += grad_output[i] * scalar;
        }
    };

    return result;
}

shared_ptr<Tensor> operator+(shared_ptr<Tensor> a, float scalar) {
    vector<float> output_data(a->getSize());
    float* input_data = a->getData();
    size_t size = a->getSize();

    for (size_t i = 0; i < size; i++) {
        output_data[i] = input_data[i] + scalar;
    }

    auto result = make_shared<Tensor>(a->getShape(), output_data, vector<shared_ptr<Tensor>>{a});

    result->_backward = [a, result]() {
        if (!a->grad) a->grad = Tensor::zeros(a->shape);

        float* grad_input = a->grad->getData();
        float* grad_output = result->grad->getData();
        size_t size = a->getSize();

        for (size_t i = 0; i < size; i++) {
            grad_input[i] += grad_output[i];
        }
    };

    return result;
}

shared_ptr<Tensor> operator-(shared_ptr<Tensor> a, shared_ptr<Tensor> b) {
    auto out = a->compute_binary_op(b, BinaryOp::SUB);
    out->parents = {a,b};

    out->_backward = [a, b, out]() {
        if (!a->grad) a->grad = Tensor::zeros(a->shape);
        if (!b->grad) b->grad = Tensor::zeros(b->shape);
        Device::get()->accumulate_grad(a, out->grad);
        Device::get()->accumulate_grad(b, out->grad * -1.0f);
    };

    return out;
}




std::shared_ptr<Tensor> operator/(std::shared_ptr<Tensor> a, float scalar) {
    return a * (1.0f / scalar);
}

std::shared_ptr<Tensor> operator/(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b){
    auto out = a->compute_binary_op(b, BinaryOp::DIV);
    out->parents = {a,b};

    out->_backward = [a, b, out]() {
        if (!a->grad) a->grad = Tensor::zeros(a->shape);
        if (!b->grad) b->grad = Tensor::zeros(b->shape);

        // dL/da = (dL/dOut) * (1/b) = grad_out / b
        auto da = out->grad / b;
        Device::get()->accumulate_grad(a, da);

        // dL/db = (dL/dOut) * (-a / b^2) 
        //       = (grad_out * -a) / (b * b)
        auto b_sq = b * b;
        auto neg_a = a * -1.0f;
        auto db = (out->grad * neg_a) / b_sq;
        Device::get()->accumulate_grad(b, db);
    };

    return out;
}


shared_ptr<Tensor> sum(shared_ptr<Tensor> a, int dim = 0) {
    auto out = a->compute_reduce_op(dim, ReduceOp::SUM);
    out->parents = {a};

    out->_backward = [a, out, dim]() {
        if (!a->grad) a->grad = Tensor::zeros(a->shape);

        float* grad_input = a->grad->getData();
        float* grad_output = out->grad->getData(); // (Batch, 1) or similar
        
        const auto& in_shape = a->shape;
        const auto& out_strides = out->strides;
        size_t total_elements = a->getSize();

        // Broadcast gradient from reduced output back to input
        for (size_t i = 0; i < total_elements; i++) {
            // Calculate input coords
            int current_idx = i;
            int out_idx = 0;
            
            for (int d = in_shape.size() - 1; d >= 0; --d) {
                int coord = current_idx % in_shape[d];
                current_idx /= in_shape[d];
                
                // If this is NOT the reduced dimension, it contributes to output index
                if (d != dim) {
                    out_idx += coord * out_strides[d];
                }
                // If it IS the reduced dimension, it is collapsed to 0 in output (so we ignore stride or add 0)
            }
            
            // 2. Accumulate gradient
            grad_input[i] += grad_output[out_idx];
        }
    };

    return out;
}

shared_ptr<Tensor> max(shared_ptr<Tensor> a, int dim = 0) {
    auto out = a->compute_reduce_op(dim, ReduceOp::MAX);
    out->parents = {a};

    out->_backward = [a, out, dim]() {
            if (!a->grad) a->grad = Tensor::zeros(a->shape);

            float* grad_in = a->grad->getData();
            float* grad_out = out->grad->getData(); // Gradiente que viene de arriba
            float* val_in = a->getData();
            float* val_out = out->getData();
            
            // Strides para mapear input -> output
            const auto& in_shape = a->shape;
            const auto& out_strides = out->strides;
            size_t total_elements = a->getSize();

            for (size_t i = 0; i < total_elements; i++) {
                // 1. Encontrar el índice en 'out' que corresponde a este elemento 'i' de 'in'
                int current_idx = i;
                int out_idx = 0;
                
                // Reconstruir coordenadas y calcular offset en output
                for (int d = in_shape.size() - 1; d >= 0; --d) {
                    int coord = current_idx % in_shape[d];
                    current_idx /= in_shape[d];
                    
                    // Si NO es la dimensión reducida, contribuye al índice del output
                    if (d != dim) {
                        out_idx += coord * out_strides[d];
                    }
                }
                
                // 2. Si este elemento fue el máximo (valor input == valor output), pasar gradiente
                if (val_in[i] == val_out[out_idx]) {
                    grad_in[i] += grad_out[out_idx];
                }
            }
        };

    return out;
}


// Transpose operation
shared_ptr<Tensor> transpose_view(shared_ptr<Tensor> a) {

    auto result = make_shared<Tensor>(*a); 
    result->grad = nullptr;
    result->parents = {a};

    int dims = a->getDimension();

    if (dims == 2) {
        // If 2D, swap rows and columns
        std::swap(result->shape[0], result->shape[1]);
        std::swap(result->strides[0], result->strides[1]);
    } else if (dims == 3) {
        // If 3D, swap the last two dimensions (rows and columns)
        std::swap(result->shape[1], result->shape[2]);
        std::swap(result->strides[1], result->strides[2]);
    } else {
        throw std::runtime_error("transpose_view: dimensiones no soportadas");
    }

    // Backward function
    result->_backward = [a, result]() {
        if (!a->grad) a->grad = Tensor::zeros(a->shape);

        // We transpose the gradient
        auto grad_T = transpose_view(result->grad);

        // Apply Device::get()->accumulate_grad to accumulate gradients correctly in case of broadcasting
        Device::get()->accumulate_grad(a, grad_T);
    };

    return result;
}




shared_ptr<Tensor> relu(shared_ptr<Tensor> a){
    
    auto out = a->compute_unary_op(UnaryOp::RELU);
    out->parents = {a};
    
    out->_backward = [a, out]() {
        if (!a->grad) a->grad = Tensor::zeros(a->shape);

        float* grad_input_data = a->grad->getData(); 
        float* grad_output_data = out->grad->getData();
        float* input_val = a->getData();

        size_t size = a->getSize();
        for(size_t i=0; i<size; i++) {
            float local_deriv = (input_val[i] > 0) ? 1.0f : 0.0f;
            grad_input_data[i] += grad_output_data[i] * local_deriv;
        }
    };

    return out;
}

shared_ptr<Tensor> exp(shared_ptr<Tensor> a){
    auto out = a->compute_unary_op(UnaryOp::EXP);
    out->parents = {a};


    out->_backward = [a,out]() {
        if(!a->grad) a->grad = Tensor::zeros(a->shape);

        float* grad_input_data = a->grad->getData();
        float* grad_output_data = out->grad->getData();
        float* out_data = out->getData();
        size_t size = out->getSize();
        for(size_t i=0; i<size;i++){
            grad_input_data[i]  += grad_output_data[i] * out_data[i];
        }
        
    };
    return out;

}

shared_ptr<Tensor> log(shared_ptr<Tensor> a){
    
    auto out = a->compute_unary_op(UnaryOp::LOG);
    out->parents = {a};


    out->_backward = [a,out]() {
        if(!a->grad) a->grad = Tensor::zeros(a->shape);

        float* grad_input_data = a->grad->getData();
        float* grad_output_data = out->grad->getData();
        float* out_data = out->getData();
        float* in_data = a->getData();
        size_t size = out->getSize();
        for(size_t i=0; i<size;i++){
            grad_input_data[i]  += grad_output_data[i] * (1.0f / (in_data[i] + 1e-8f));
        }
        
    };
    return out;
}

shared_ptr<Tensor> sigmoid(shared_ptr<Tensor> a){
    auto out = a->compute_unary_op(UnaryOp::SIGMOID);
    out->parents = {a};

    out->_backward = [a, out]() {
        if (!a->grad) a->grad = Tensor::zeros(a->shape);

        float* grad_input_data = a->grad->getData(); 
        float* grad_output_data = out->grad->getData();
        float* input_val = a->getData();
        float* output_val = out->getData();

        size_t size = a->getSize();
        for(size_t i=0; i<size; i++) {
            float local_deriv = output_val[i] * (1.0f - output_val[i]);
            grad_input_data[i] += grad_output_data[i] * local_deriv;
        }
    };

    return out;

}

shared_ptr<Tensor> tanh(shared_ptr<Tensor> a){
    auto out = a->compute_unary_op(UnaryOp::TANH);
    out->parents = {a};
        
    out->_backward = [a, out]() {
        if (!a->grad) a->grad = Tensor::zeros(a->shape);

        float* grad_input_data = a->grad->getData(); 
        float* grad_output_data = out->grad->getData();
        float* output_val = out->getData();

        size_t size = a->getSize();
        for(size_t i=0; i<size; i++) {
            float local_deriv = 1.0f - (output_val[i] * output_val[i]);
            grad_input_data[i] += grad_output_data[i] * local_deriv;
        }
    };

    return out;
}




