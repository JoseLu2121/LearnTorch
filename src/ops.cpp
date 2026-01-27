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
    // Create the out tensor
    auto out = std::make_shared<Tensor>(std::vector<int>{batch_out, a_view->shape[1],b_view->shape[2]});
    out->parents = {a, b}; 

    TensorInfo i_a = a_view.get()->getInfo();
    TensorInfo i_b = b_view.get()->getInfo();
    TensorInfo i_out = out.get()->getInfo();
    // Backend gemm call
    static CPUBackend backend;
    backend.gemm(i_a,i_b, i_out);
    

    out->_backward = [a, b, out]() {
        if (!a->grad) a->grad = Tensor::zeros(a->shape);
        if (!b->grad) b->grad = Tensor::zeros(b->shape);
        auto grad_output = out->grad;
        
        // 1. Gradiente respecto a A (Input): dA = Grad @ B.T
        // Matriz (Batch, Out) @ (Out, In) -> (Batch, In)
        auto b_T = transpose_view(b);
        auto da = matmul(grad_output, b_T,false);
        Device::get()->accumulate_grad(a, da);

        // 2. Gradiente respecto a B (Pesos): dB = A.T @ Grad
        // Matriz (In, Batch) @ (Batch, Out) -> (In, Out)
        auto a_T = transpose_view(a);
        auto db = matmul(a_T, grad_output, false);
        
        // Device::get()->accumulate_grad will now handle transposed accumulation correctly
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
    // Forward: out = a + scalar
    // Copiamos datos y sumamos
    vector<float> output_data(a->getSize());
    float* input_data = a->getData();
    size_t size = a->getSize();

    for (size_t i = 0; i < size; i++) {
        output_data[i] = input_data[i] + scalar;
    }

    auto result = make_shared<Tensor>(a->getShape(), output_data, vector<shared_ptr<Tensor>>{a});

    // Backward: d(x+c)/dx = 1
    // Simplemente pasamos el gradiente hacia atrás tal cual
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
    auto out = a->compute_binary_op(b, BinaryOp::ADD);
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


shared_ptr<Tensor> sum(shared_ptr<Tensor> a) {
    float total = 0.0f;
    float* data = a->getData();
    size_t size = a->getSize(); 

    for(size_t i = 0; i < size; i++) {
        total += data[i];
    }

    vector<float> res_data = {total};
    auto result = make_shared<Tensor>(vector<int>{1}, res_data , vector<shared_ptr<Tensor>>{a});


    result->_backward = [a, result]() {
        if (!a->grad) a->grad = Tensor::zeros(a->shape);

        float grad_output = result->grad->getData()[0];
        
        float* grad_input = a->grad->getData();
        size_t n = a->getSize();

        for(size_t i = 0; i < n; i++) {
            grad_input[i] += grad_output;
        }
    };

    return result;
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
    
    auto out = std::make_shared<Tensor>(a->shape);
    out->parents = {a};
    TensorInfo i_out = out.get()->getInfo();

    Device::get()->unary(a.get()->getInfo(), i_out, UnaryOp::RELU);
    
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
    auto out = std::make_shared<Tensor>(a->shape);
    out->parents = {a};
    TensorInfo i_out = out.get()->getInfo();

    Device::get()->unary(a.get()->getInfo(), i_out, UnaryOp::EXP);

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
    
     auto out = std::make_shared<Tensor>(a->shape);
    out->parents = {a};
    TensorInfo i_out = out.get()->getInfo();

    Device::get()->unary(a.get()->getInfo(), i_out, UnaryOp::LOG);

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
    auto out = std::make_shared<Tensor>(a->shape);
    out->parents = {a};
    TensorInfo i_out = out.get()->getInfo();

    Device::get()->unary(a.get()->getInfo(), i_out, UnaryOp::SIGMOID);
        
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
    auto out = std::make_shared<Tensor>(a->shape);
    out->parents = {a};
    TensorInfo i_out = out.get()->getInfo();

    Device::get()->unary(a.get()->getInfo(), i_out, UnaryOp::TANH);
        
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




