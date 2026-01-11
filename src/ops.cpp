#include "ops.h"
#include "utils.h"
#include <algorithm>
#include <stdexcept>
#include <cmath> 
#include <iostream> // Necesario para cout

using namespace std;

// Unbroadcast function to accumulate gradients correctly
void unbroadcast(shared_ptr<Tensor> param, shared_ptr<Tensor> incoming_grad) {

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




Tensor dot_scalar_product(shared_ptr<Tensor> a, shared_ptr<Tensor> b) {
    int size_output = a->getShape().at(0);
    vector<float> output_data(size_output);
    

    int stride_a = a->strides[0];
    int stride_b = b->strides[0];

    for (int i = 0; i < size_output; i++) {

        float val_a = a->getData()[i * stride_a];
        float val_b = b->getData()[i * stride_b];
        output_data[i] = val_a * val_b;
    }
    return Tensor({size_output}, output_data , {a,b});
}


Tensor vector_matrix_product(shared_ptr<Tensor> v, shared_ptr<Tensor> m) {
    int column_v = v->getShape().at(1);
    int column_m = m->getShape().at(1);
    

    int stride_v_col = v->strides[1]; 
    int stride_m_row = m->strides[0]; 
    int stride_m_col = m->strides[1]; 

    vector<float> output_data(column_m);
    
    for (int i = 0; i < column_m; i++) {
        float sum = 0;
        for (int j = 0; j < column_v; j++) {
            // v: índice j * su stride
            // m: fila j (j*stride_row) + columna i (i*stride_col)
            float val_v = v->getData()[j * stride_v_col];
            float val_m = m->getData()[j * stride_m_row + i * stride_m_col];
            sum += val_v * val_m;
        }
        output_data[i] = sum;
    }
    return Tensor({column_m}, output_data , {v,m});
}

Tensor matrix_matrix_product(shared_ptr<Tensor> m, shared_ptr<Tensor> v) {
    int row_m = m->shape.at(0);
    int col_v = v->shape.at(1);
    int col_m = m->shape.at(1);
    
    vector<int> output_shape = {row_m, col_v};
    vector<float> output_data(element_vector_product(output_shape));
    

    int m_stride_0 = m->strides[0];
    int m_stride_1 = m->strides[1];
    int v_stride_0 = v->strides[0];
    int v_stride_1 = v->strides[1];

    for (int i = 0; i < row_m; i++) {
        for (int j = 0; j < col_v; j++) {
            float sum = 0;
            for (int k = 0; k < col_m; k++) {
                // M[i, k] * V[k, j]
                float val_m = m->getData()[i * m_stride_0 + k * m_stride_1];
                float val_v = v->getData()[k * v_stride_0 + j * v_stride_1];
                sum += val_m * val_v;
            }
            output_data[i * col_v + j] = sum;
        }
    }
    
    return Tensor(output_shape, output_data , {m,v});
}

Tensor batch_matrix_product(shared_ptr<Tensor> b, shared_ptr<Tensor> m) {
    int b_batch = b->getShape().at(0);
    int m_col = m->getShape().at(1);
    int b_row = b->getShape().at(1);

    vector<int> output_shape = {b_batch, b_row,m_col};

    vector<float> output_data(element_vector_product(output_shape));
    for (int i = 0; i < b_batch; i++) {
        auto matrix_batch_i = b->getBatch(i);
        Tensor matrix_output = matrix_matrix_product(matrix_batch_i, m);
        float* matrix_data = matrix_output.getData();
        copy(matrix_data, matrix_data + b_row * m_col, output_data.begin() + b_row * m_col * i);
    }
    return Tensor(output_shape, output_data , {b,m});
}

Tensor vector_batch_product(shared_ptr<Tensor> v, shared_ptr<Tensor> b){
    auto v_view = make_shared<Tensor>(*v);
    v_view->shape.insert(v_view->shape.begin(),1);
    v_view->strides.insert(v_view->strides.begin(),0);
    int b_batch = b->shape.at(0);
    int b_col = b->shape.at(2);
    vector<int> output_shape = {b_batch,b_col};
    vector<float> output_data(element_vector_product(output_shape));
    for(int i = 0; i < b_batch; i++){
        auto matrix_batch_i = b->getBatch(i);
        Tensor matrix_output = vector_matrix_product(v_view, matrix_batch_i);
        float* matrix_data = matrix_output.getData();
        copy(matrix_data, matrix_data + b_col, output_data.begin() + i * b_col);
    }
    return Tensor(output_shape, output_data , {v,b});
 }

Tensor matrix_batch_product(shared_ptr<Tensor> m, shared_ptr<Tensor> b) {
    int b_batch = b->getShape().at(0);
    int m_row = m->getShape().at(0);
    int b_col = b->getShape().at(2);
    vector<int> output_shape = {b_batch, m_row, b_col};
    vector<float> output_data(element_vector_product(output_shape));
    for (int i = 0; i < b_batch; i++) {
        auto matrix_batch_i = b->getBatch(i);
        Tensor matrix_output = matrix_matrix_product(m, matrix_batch_i);
        float* matrix_data = matrix_output.getData();
        copy(matrix_data, matrix_data + m_row*b_col, output_data.begin() + m_row*b_col* i);
    }
    return Tensor(output_shape, output_data , {m,b});
}


Tensor batch_batch_product(shared_ptr<Tensor> a, shared_ptr<Tensor> b) {
    int batch = a->shape[0]; 

    int m = a->shape[1];
    int n = b->shape[2];
    
    vector<int> output_shape = {batch, m, n};
    vector<float> output_data(element_vector_product(output_shape));

    for(int i=0; i<batch; i++){

        int idx_a = (i < a->shape[0]) ? i : 0;
        int idx_b = (i < b->shape[0]) ? i : 0;

        auto sub_a = a->getBatch(idx_a);
        auto sub_b = b->getBatch(idx_b);

        auto sub_res = matrix_matrix_product(sub_a, sub_b);
 
        float* src = sub_res.getData();
        float* dst = output_data.data()  + i * (m*n);
        copy(src, src + (m*n), dst);
    }
    return Tensor(output_shape, output_data , {a,b});
}

shared_ptr<Tensor> operator+(shared_ptr<Tensor> a, shared_ptr<Tensor> b) {

    auto a_view = a->view_to_3d();
    auto b_view = b->view_to_3d();

    int out_batch = max(a_view->shape[0], b_view->shape[0]);
    int out_rows  = max(a_view->shape[1], b_view->shape[1]);
    int out_cols  = max(a_view->shape[2], b_view->shape[2]);
    vector<int> output_shape = {out_batch, out_rows, out_cols};
    
    vector<int> sA = a_view->strides;
    if (a_view->shape[0] == 1 && out_batch > 1) sA[0] = 0;
    if (a_view->shape[1] == 1 && out_rows  > 1) sA[1] = 0;
    if (a_view->shape[2] == 1 && out_cols  > 1) sA[2] = 0;

    vector<int> sB = b_view->strides;
    if (b_view->shape[0] == 1 && out_batch > 1) sB[0] = 0;
    if (b_view->shape[1] == 1 && out_rows  > 1) sB[1] = 0;
    if (b_view->shape[2] == 1 && out_cols  > 1) sB[2] = 0;

    vector<float> output_data(element_vector_product(output_shape));
    float* data_a = a_view->getData();
    float* data_b = b_view->getData();

    for (int k = 0; k < out_batch; k++) {
        for (int i = 0; i < out_rows; i++) {
            for (int j = 0; j < out_cols; j++) {
                int idx_a = k * sA[0] + i * sA[1] + j * sA[2];
                int idx_b = k * sB[0] + i * sB[1] + j * sB[2];
                int idx_out = k * (out_rows * out_cols) + i * out_cols + j;
                output_data[idx_out] = data_a[idx_a] + data_b[idx_b];
            }
        }
    }

    auto result = make_shared<Tensor>(output_shape, output_data , vector<shared_ptr<Tensor>>{a, b});


    result->_backward = [a, b, result]() {

        if (!a->grad) a->grad = Tensor::zeros(a->shape);
        if (!b->grad) b->grad = Tensor::zeros(b->shape);


        unbroadcast(a, result->grad);
        unbroadcast(b, result->grad);
    };

    return result;
}


float relu_function(float x){
    if(x<0) x=0;
    return x;
}

shared_ptr<Tensor> relu(shared_ptr<Tensor> a){
    vector<float> output_data(element_vector_product(a->getShape()));
    for(size_t i=0; i< a->getSize();i++){
        output_data[i] = relu_function(a->getData()[i]); 
    }
    
    auto result = make_shared<Tensor>(a->getShape(), output_data , vector<shared_ptr<Tensor>>{a});

    result->_backward = [a, result]() {
        if (!a->grad) a->grad = Tensor::zeros(a->shape);

        float* grad_input_data = a->grad->getData(); 
        float* grad_output_data = result->grad->getData();
        float* input_val = a->getData();

        size_t size = a->getSize();
        for(size_t i=0; i<size; i++) {
            float local_deriv = (input_val[i] > 0) ? 1.0f : 0.0f;
            grad_input_data[i] += grad_output_data[i] * local_deriv;
        }
    };

    return result;
}


shared_ptr<Tensor> matmul(shared_ptr<Tensor> a, shared_ptr<Tensor> b, bool require_grad) {
    shared_ptr<Tensor> result;

    if (a->getDimension() == 1 && b->getDimension() == 1) result = make_shared<Tensor> (dot_scalar_product(a, b));
    else if (a->getDimension() == 1 && b->getDimension() == 2) {
        auto a_view = make_shared<Tensor>(*a);
        a_view->shape.insert(a_view->shape.begin(), 1);
        a_view->strides.insert(a_view->strides.begin(), 0);
        result = make_shared<Tensor> (vector_matrix_product(a_view, b));
    }
    else if (a->getDimension() == 2 && b->getDimension() == 1) {
        auto b_view = make_shared<Tensor>(*b);
        b_view->shape.push_back(1);
        b_view->strides.push_back(0);
        result = make_shared<Tensor> (matrix_matrix_product(a, b_view));
    }
    else if (a->getDimension() == 2 && b->getDimension() == 2) result = make_shared<Tensor> (matrix_matrix_product(a, b));
    else if (a->getDimension() == 3 && b->getDimension() == 2) result = make_shared<Tensor> (batch_matrix_product(a, b));
    else if (a->getDimension() == 3 && b->getDimension() == 1) {
        auto b_view = make_shared<Tensor>(*b);
        b_view->shape.push_back(1);
        b_view->strides.push_back(0);
        result = make_shared<Tensor> (batch_matrix_product(a, b_view));
    }
    else if(a->getDimension() == 1 && b->getDimension() == 3){
        result = make_shared<Tensor> (vector_batch_product(a,b));
    }
    else if(a->getDimension()==2 && b->getDimension() == 3){
        result = make_shared<Tensor>(matrix_batch_product(a,b));
    }
    else if(a->getDimension() == 3 && b->getDimension() == 3) {
        result = make_shared<Tensor>(batch_batch_product(a, b));
    }
    else throw runtime_error("Dimensiones no soportadas");

    result->parents = {a, b}; 

        result->_backward = [a, b, result]() {
        if (!a->grad) a->grad = Tensor::zeros(a->shape);
        if (!b->grad) b->grad = Tensor::zeros(b->shape);
        auto grad_output = result->grad;
        
        // 1. Gradiente respecto a A (Input): dA = Grad @ B.T
        // Matriz (Batch, Out) @ (Out, In) -> (Batch, In)
        auto b_T = transpose_view(b);
        auto da = matmul(grad_output, b_T,false);
        unbroadcast(a, da);

        // 2. Gradiente respecto a B (Pesos): dB = A.T @ Grad
        // Matriz (In, Batch) @ (Batch, Out) -> (In, Out)
        auto a_T = transpose_view(a);
        auto db = matmul(a_T, grad_output, false);
        
        // Unbroadcast will now handle transposed accumulation correctly
        unbroadcast(b, db);
    };

    return result;
}


std::shared_ptr<Tensor> operator*(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) { 
    if (a->getSize() != b->getSize()) { throw std::runtime_error("Element-wise mul: Tamaños no coinciden"); } 
    std::vector<float> output_data(a->getSize()); 
    float* data_a = a->getData(); 
    float* data_b = b->getData(); 
    size_t size = a->getSize();
     // Forward: a[i] * b[i] 
    for (size_t i = 0; i < size; i++) { 
        output_data[i] = data_a[i] * data_b[i];
    } 
    auto result = std::make_shared<Tensor>(a->getShape(), output_data, 
        std::vector<std::shared_ptr<Tensor>>{a, b}); 
    // Backward: Product Rule // si y = a * b // dy/da = b * dy // dy/db = a * dy 
    result->_backward = [a, b, result]() { 
        if (!a->grad) a->grad = Tensor::zeros(a->shape);
        if (!b->grad) b->grad = Tensor::zeros(b->shape); 

        float* grad_out = result->grad->getData(); 
        float* grad_a = a->grad->getData(); 
        float* grad_b = b->grad->getData(); 
        float* val_a = a->getData(); 
        float* val_b = b->getData(); 
        size_t sz = a->getSize(); 
        for (size_t i = 0; i < sz; i++) { 
            grad_a[i] += val_b[i] * grad_out[i]; 
            grad_b[i] += val_a[i] * grad_out[i]; 
            } 
        };
        return result;
    
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


shared_ptr<Tensor> operator-(shared_ptr<Tensor> a, shared_ptr<Tensor> b) {
    return a + (b * -1.0f);
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



std::shared_ptr<Tensor> linear_activation(std::shared_ptr<Tensor> x) {
    return x;
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

        // Apply unbroadcast to accumulate gradients correctly in case of broadcasting
        unbroadcast(a, grad_T);
    };

    return result;
}








