#include "ops.h"
#include "types.h"
#include "utils.h"
#include "backend.h"
#include "device.h"
#include <algorithm>
#include <stdexcept>
#include <cmath> 
#include <iostream> 

using namespace std;


std::shared_ptr<Tensor> linear_activation(std::shared_ptr<Tensor> x) {
    return x; 
}

shared_ptr<Tensor> operator+(shared_ptr<Tensor> a, shared_ptr<Tensor> b) {
    auto out = a->compute_binary_op(b, BinaryOp::ADD);

    out->parents = {a,b};
    Tensor* raw_out = out.get();

    out->_backward = [a, b, raw_out]() {
        if (!a->grad) a->grad = Tensor::zeros(a->shape);
        if (!b->grad) b->grad = Tensor::zeros(b->shape);
        Device::get()->accumulate_grad(a, raw_out->grad);
        Device::get()->accumulate_grad(b, raw_out->grad);
    };

    return out;
}

shared_ptr<Tensor> matmul(shared_ptr<Tensor> a, shared_ptr<Tensor> b, bool require_grad) {
    auto a_view = a->view_to_gemm(false);
    auto b_view = b->view_to_gemm(true);
    int batch_out = std::max(a_view->shape[0], b_view->shape[0]);
    
    std::vector<int> out_shape;
    if (a->shape.size() == 2 && b->shape.size() == 2) {
        out_shape = {a_view->shape[1], b_view->shape[2]};
    } else {
        out_shape = {batch_out, a_view->shape[1], b_view->shape[2]};
    }

    auto out = std::make_shared<Tensor>(out_shape);
    out->parents = {a, b}; 

    shared_ptr<Tensor> out_view;
    if (out_shape.size() == 2) {
       out_view = out->view_to_gemm(false); 
    } else {
       out_view = out; 
    }

    TensorInfo i_a = a_view.get()->getInfo();
    TensorInfo i_b = b_view.get()->getInfo();
    TensorInfo i_out = out_view.get()->getInfo();
    
    static CPUBackend backend;
    backend.gemm(i_a,i_b, i_out);

    Tensor* raw_out = out.get();

    out->_backward = [a, b, raw_out]() {
        if (!a->grad) a->grad = Tensor::zeros(a->shape);
        if (!b->grad) b->grad = Tensor::zeros(b->shape);
        auto grad_output = raw_out->grad;
        
        auto b_T = transpose_view(b);
        auto da = matmul(grad_output, b_T,false);
        Device::get()->accumulate_grad(a, da);

        auto a_T = transpose_view(a);
        auto db = matmul(a_T, grad_output, false);
        
        Device::get()->accumulate_grad(b, db);
    };

    return out;
}


std::shared_ptr<Tensor> operator*(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) { 
    
    auto out = a->compute_binary_op(b, BinaryOp::MUL);

    out->parents = {a,b};
    Tensor* raw_out = out.get();

    out->_backward = [a, b, raw_out]() { 
        if (!a->grad) a->grad = Tensor::zeros(a->shape);
        if (!b->grad) b->grad = Tensor::zeros(b->shape); 

        auto grad_a_part = raw_out->grad * b; 
        Device::get()->accumulate_grad(a, grad_a_part);

        auto grad_b_part = raw_out->grad * a;
        Device::get()->accumulate_grad(b, grad_b_part);

        };

    return out;
    
}

shared_ptr<Tensor> operator*(shared_ptr<Tensor> a, float scalar) {
    vector<float> output_data(element_vector_product(a->getShape()));
    float* input_data = a->getData();
    size_t size = a->getSize();

    for (size_t i = 0; i < size; i++) {
        output_data[i] = input_data[i] * scalar;
    }

    auto result = make_shared<Tensor>(a->getShape(), output_data , vector<shared_ptr<Tensor>>{a});
    Tensor* raw_result = result.get();

    result->_backward = [a, raw_result, scalar]() {
        if (!a->grad) a->grad = Tensor::zeros(a->shape);

        float* grad_input = a->grad->getData();
        float* grad_output = raw_result->grad->getData();
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
    Tensor* raw_result = result.get();

    result->_backward = [a, raw_result]() {
        if (!a->grad) a->grad = Tensor::zeros(a->shape);

        float* grad_input = a->grad->getData();
        float* grad_output = raw_result->grad->getData();
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
    Tensor* raw_out = out.get();

    out->_backward = [a, b, raw_out]() {
        if (!a->grad) a->grad = Tensor::zeros(a->shape);
        if (!b->grad) b->grad = Tensor::zeros(b->shape);
        Device::get()->accumulate_grad(a, raw_out->grad);
        Device::get()->accumulate_grad(b, raw_out->grad * -1.0f);
    };

    return out;
}

std::shared_ptr<Tensor> operator/(std::shared_ptr<Tensor> a, float scalar) {
    return a * (1.0f / scalar);
}

std::shared_ptr<Tensor> operator/(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b){
    auto out = a->compute_binary_op(b, BinaryOp::DIV);
    out->parents = {a,b};
    Tensor* raw_out = out.get();

    out->_backward = [a, b, raw_out]() {
        if (!a->grad) a->grad = Tensor::zeros(a->shape);
        if (!b->grad) b->grad = Tensor::zeros(b->shape);

        auto da = raw_out->grad / b;
        Device::get()->accumulate_grad(a, da);

        auto b_sq = b * b;
        auto neg_a = a * -1.0f;
        auto db = (raw_out->grad * neg_a) / b_sq;
        Device::get()->accumulate_grad(b, db);
    };

    return out;
}


shared_ptr<Tensor> sum(shared_ptr<Tensor> a, int dim = 0) {
    auto out = a->compute_reduce_op(dim, ReduceOp::SUM);
    out->parents = {a};
    Tensor* raw_out = out.get();

    out->_backward = [a, raw_out, dim]() {
        if (!a->grad) a->grad = Tensor::zeros(a->shape);

        float* grad_input = a->grad->getData();
        float* grad_output = raw_out->grad->getData(); 
        
        const auto& in_shape = a->shape;
        const auto& out_strides = raw_out->strides;
        size_t total_elements = a->getSize();

        for (size_t i = 0; i < total_elements; i++) {
            int current_idx = i;
            int out_idx = 0;
            
            for (int d = in_shape.size() - 1; d >= 0; --d) {
                int coord = current_idx % in_shape[d];
                current_idx /= in_shape[d];
                
                if (d != dim) {
                    out_idx += coord * out_strides[d];
                }
            }
            
            grad_input[i] += grad_output[out_idx];
        }
    };

    return out;
}

shared_ptr<Tensor> max(shared_ptr<Tensor> a, int dim = 0) {
    auto out = a->compute_reduce_op(dim, ReduceOp::MAX);
    out->parents = {a};
    Tensor* raw_out = out.get();

    out->_backward = [a, raw_out, dim]() {
            if (!a->grad) a->grad = Tensor::zeros(a->shape);

            float* grad_in = a->grad->getData();
            float* grad_out = raw_out->grad->getData(); 
            float* val_in = a->getData();
            float* val_out = raw_out->getData();
            
            const auto& in_shape = a->shape;
            const auto& out_strides = raw_out->strides;
            size_t total_elements = a->getSize();

            for (size_t i = 0; i < total_elements; i++) {
                int current_idx = i;
                int out_idx = 0;
                
                for (int d = in_shape.size() - 1; d >= 0; --d) {
                    int coord = current_idx % in_shape[d];
                    current_idx /= in_shape[d];
                    
                    if (d != dim) {
                        out_idx += coord * out_strides[d];
                    }
                }
                
                if (val_in[i] == val_out[out_idx]) {
                    grad_in[i] += grad_out[out_idx];
                }
            }
        };

    return out;
}

shared_ptr<Tensor> argmax(shared_ptr<Tensor> a, int dim =0){
    auto out = a->compute_reduce_op(dim, ReduceOp::ARGMAX);
    out->parents = {a};
    Tensor* raw_out = out.get();
    
    out->_backward = [a,raw_out]() {
        (void) a; (void) raw_out;
    };

    return out;
}


shared_ptr<Tensor> transpose_view(shared_ptr<Tensor> a) {

    auto result = make_shared<Tensor>(*a); 
    result->grad = nullptr;
    result->parents = {a};

    int dims = a->getDimension();

    if (dims == 2) {
        std::swap(result->shape[0], result->shape[1]);
        std::swap(result->strides[0], result->strides[1]);
    } else if (dims == 3) {
        std::swap(result->shape[1], result->shape[2]);
        std::swap(result->strides[1], result->strides[2]);
    } else {
        throw std::runtime_error("transpose_view: dimensiones no soportadas");
    }

    Tensor* raw_result = result.get();

    result->_backward = [a, raw_result]() {
        if (!a->grad) a->grad = Tensor::zeros(a->shape);

        auto grad_T = transpose_view(raw_result->grad);

        Device::get()->accumulate_grad(a, grad_T);
    };

    return result;
}

shared_ptr<Tensor> relu(shared_ptr<Tensor> a){
    
    auto out = a->compute_unary_op(UnaryOp::RELU);
    out->parents = {a};
    Tensor* raw_out = out.get();
    
    out->_backward = [a, raw_out]() {
        if (!a->grad) a->grad = Tensor::zeros(a->shape);

        float* grad_input_data = a->grad->getData(); 
        float* grad_output_data = raw_out->grad->getData();
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
    Tensor* raw_out = out.get();

    out->_backward = [a,raw_out]() {
        if(!a->grad) a->grad = Tensor::zeros(a->shape);

        float* grad_input_data = a->grad->getData();
        float* grad_output_data = raw_out->grad->getData();
        float* out_data = raw_out->getData();
        size_t size = raw_out->getSize();
        for(size_t i=0; i<size;i++){
            grad_input_data[i]  += grad_output_data[i] * out_data[i];
        }
        
    };
    return out;

}

shared_ptr<Tensor> sqrt(shared_ptr<Tensor> a) {
    auto out = a->compute_unary_op(UnaryOp::SQRT);
    out->parents = {a};
    Tensor* raw_out = out.get();

    out->_backward = [a, raw_out]() {
        if(!a->grad) {
            a->grad = Tensor::zeros(a->shape);
        }
        auto out_shared = raw_out->shared_from_this();

        auto grad_sqrt = (raw_out->grad * 0.5f) / out_shared;

        Device::get()->accumulate_grad(a, grad_sqrt);

    };

    return out;

}

shared_ptr<Tensor> log(shared_ptr<Tensor> a){
    
    auto out = a->compute_unary_op(UnaryOp::LOG);
    out->parents = {a};
    Tensor* raw_out = out.get();

    out->_backward = [a,raw_out]() {
        if(!a->grad) a->grad = Tensor::zeros(a->shape);

        float* grad_input_data = a->grad->getData();
        float* grad_output_data = raw_out->grad->getData();
        float* out_data = raw_out->getData();
        float* in_data = a->getData();
        size_t size = raw_out->getSize();
        for(size_t i=0; i<size;i++){
            grad_input_data[i]  += grad_output_data[i] * (1.0f / (in_data[i] + 1e-8f));
        }
        
    };
    return out;
}

shared_ptr<Tensor> sigmoid(shared_ptr<Tensor> a){
    auto out = a->compute_unary_op(UnaryOp::SIGMOID);
    out->parents = {a};
    Tensor* raw_out = out.get();

    out->_backward = [a, raw_out]() {
        if (!a->grad) a->grad = Tensor::zeros(a->shape);

        float* grad_input_data = a->grad->getData(); 
        float* grad_output_data = raw_out->grad->getData();
        float* input_val = a->getData();
        float* output_val = raw_out->getData();

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
    Tensor* raw_out = out.get();
        
    out->_backward = [a, raw_out]() {
        if (!a->grad) a->grad = Tensor::zeros(a->shape);

        float* grad_input_data = a->grad->getData(); 
        float* grad_output_data = raw_out->grad->getData();
        float* output_val = raw_out->getData();

        size_t size = a->getSize();
        for(size_t i=0; i<size; i++) {
            float local_deriv = 1.0f - (output_val[i] * output_val[i]);
            grad_input_data[i] += grad_output_data[i] * local_deriv;
        }
    };

    return out;
}

shared_ptr<Tensor> gather(shared_ptr<Tensor> w, shared_ptr<Tensor> ind) {
    auto out = w->compute_gather(ind);

    out->parents = {w};
    Tensor* raw_out = out.get();

    out->_backward = [w, ind, raw_out] {

        if(!w->grad) {
            w->grad = Tensor::zeros(w->shape);
        }

        w->grad->compute_scatter_add(ind, raw_out->grad);
    };


    return out;
}