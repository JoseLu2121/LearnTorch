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

    auto out = a->compute_matmul(b);
    out->parents = {a, b}; 

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

std::shared_ptr<Tensor> conv2d(std::shared_ptr<Tensor>  input, std::shared_ptr<Tensor>  w, int stride, int padding) {
        auto out = input->compute_conv2d(w,stride,padding);
        out->parents = {input, w};
        Tensor* raw_out = out.get();


        out->_backward = [input, w, raw_out, stride, padding]() {
            if (!input->grad) input->grad = Tensor::zeros(input->shape);
            if (!w->grad) w->grad = Tensor::zeros(w->shape);

            int n_batch = input->shape[0];
            int c_in = input->shape[1];
            int h_in = input->shape[2];
            int w_in = input->shape[3];

            int c_out = w->shape[0];
            int k_h = w->shape[2];
            int k_w = w->shape[3];

            int h_out = raw_out->shape[2];
            int w_out = raw_out->shape[3];

            int workspace_size = c_in * k_h * k_w * h_out * w_out;
            float* col_buffer = new float[workspace_size];
            
            std::vector<float> w_data(w->getData(), w->getData() + w->getSize());
            auto w_2d = std::make_shared<Tensor>(std::vector<int>{c_out, c_in * k_h * k_w}, w_data);
            auto w_T = transpose_view(w_2d);
            cout << "Entramos backward conv2d" << endl;
            for(int b = 0; b < n_batch; b++) {
                auto x_b = input->batch_view(b, true);
                auto dY_b = raw_out->grad->batch_view(b, true);
                auto dx_b = input->grad->batch_view(b,true);

                std::vector<float> dy_data(dY_b->getData(), dY_b->getData() + dY_b->getSize());
                auto dY_b_2d = std::make_shared<Tensor>(std::vector<int>{c_out, h_out * w_out}, dy_data);

                x_b->im2col(col_buffer, h_out, w_out, k_h, k_w, stride, padding);

                std::vector<float> col_data(col_buffer, col_buffer + workspace_size);
                auto X_col = std::make_shared<Tensor>(std::vector<int>{c_in * k_h * k_w, h_out * w_out}, col_data);
                X_col->strides = {1, c_in * k_h * k_w};

                auto X_col_T = transpose_view(X_col);
                auto dW_b = matmul(dY_b_2d, X_col_T, false);

                dW_b->shape = w->shape;
                dW_b->strides = w->strides;
                Device::get()->accumulate_grad(w, dW_b);

                auto dX_col = matmul(w_T, dY_b_2d, false);
                dx_b->col2im(dX_col->getData(), h_out, w_out, k_h, k_w, stride, padding);
            }
            
            delete[] col_buffer;
            cout << "Salimos backward conv2d" << endl;
        };

        return out;

    }

shared_ptr<Tensor> flatten(shared_ptr<Tensor> a) {
    auto out = a->compute_flatten();
    
    out->parents = {a};
    Tensor* raw_out = out.get();
    
    out->_backward = [a, raw_out]() {
        if (!a->grad) a->grad = Tensor::zeros(a->shape);
        
        float* grad_in = a->grad->getData();
        float* grad_out = raw_out->grad->getData();
        
        for (size_t i = 0; i < a->getSize(); i++) {
            grad_in[i] += grad_out[i];
        }
    };
    
    return out;
}


