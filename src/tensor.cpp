#include "tensor.h"
#include "utils.h" 
#include "types.h"
#include "device.h"
#include "backend.h"
#include <fstream>
#include <stdexcept>

using namespace std;

// ====================
// Constructors
// ====================
Tensor::Tensor(const vector<int>& shape_param, const vector<float>& data_param,
       const vector<shared_ptr<Tensor>> parents_param) 
      // Setting the params given to the attributes of the tensor
    : data(nullptr), 
      shape(shape_param), 
      total_size(element_vector_product(shape_param)), 
      parents(parents_param) {
    
    // If the size is 0 we return an empty tensor
    if (total_size == 0) return;

    // The data is a pointer to a float array with tensor size
    this->data = shared_ptr<float[]>(new float[total_size]);
    
    // We initialize the data with the param given
    if (!data_param.empty()) {
        if (data_param.size() != total_size) {
            throw std::invalid_argument("The data given doesn't match the shape of the tensor");
        }


        for (size_t i = 0; i < total_size; i++) {
            data[i] = data_param[i];
        }
    }else{
        // If we have no data given, we set the tensor data to zero
        for (size_t i = 0; i < total_size; i++) {
            data[i] = 0.0f;
        }
    }

    // We initialize the Strides vector
    strides.resize(shape.size());
    // The last one is always 1
    strides[shape.size() - 1] = 1;
    // Each element is himself times the previous element from the right
    for (int i = shape.size() - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
}

// Copy or view constructor
Tensor::Tensor(const Tensor& other) 
    : data(other.data),       
      shape(other.shape),     
      strides(other.strides), 
      total_size(other.total_size),
      grad(nullptr), // We avoid issues with the original grad tensor
      parents({}){}


// ====================
// Utils for debugging
// ====================

void Tensor::printElements(int count) const {
    cout << "Elementos del tensor" << endl;
    for (int i = 0; i < count; i++) {

        cout << "Elemento " << i << ": " << getData()[i] << endl;
    }
}

void Tensor::printShape() const{ 
    cout << "Shape: ("; 
    for (size_t i = 0; i < shape.size(); i++) {
        cout << shape[i];
        if (i != shape.size() - 1) cout << ", ";
    }
    cout << ")" << endl;
}

void Tensor::printStrides() const { 
    cout << "Strides: ("; 
    for (size_t i = 0; i < strides.size(); i++) {
        cout << strides[i];
        if (i != strides.size() - 1) cout << ", ";
    }
    cout << ")" << endl;
}

void Tensor::info(int max_size) const {
    cout << "Tensor Info:" << endl;
    cout << "  Shape: (";
    for (size_t i = 0; i < shape.size(); i++) 
        cout << shape[i] << (i < shape.size() - 1 ? ", " : "");
    cout << ")" << endl;
    
    cout << "  Strides: (";
    for (size_t i = 0; i < strides.size(); i++) 
        cout << strides[i] << (i < strides.size() - 1 ? ", " : "");
    cout << ")" << endl;

    // We only print the elements if the tensor is small 
    if (total_size <= max_size) {
        cout << "  Data: [ ";
        for (int i = 0; i < max_size; i++) cout << data[i] << " ";
        cout << "]" << endl;
    } else {
        cout << "  Data: [ ... " << total_size << " elementos ... ]" << endl;
    }
    cout << "-------------------------" << endl;
}

// ===================
// Size and view manipulation
// ===================

shared_ptr<Tensor> Tensor::view_to_3d() {
    // We create a view of the tensor
    auto view = make_shared<Tensor>(*this); 

    if(view->getDimension() == 1){
        view->strides.push_back(0); 
        view->shape.push_back(1);
    }
    if(view->getDimension() == 2){
        view->strides.insert(view->strides.begin(), 0); 
        view->shape.insert(view->shape.begin(), 1);
    }
    return view;
}

shared_ptr<Tensor> Tensor::view_to_gemm(bool as_b_term) {
    
    auto view = make_shared<Tensor>(*this);
    // Same logic as view_to_3d, but depends on whether the tensor is the second operand
    if(view->getDimension() == 1){
        if(as_b_term){
            view->shape = {1, view->shape[0], 1};
            view->strides = {0, view->strides[0], 0};
        } else{
            view->shape = {1,1,view->shape[0]};
            view->strides = {0,0, view->strides[0],};
        }
    }
    else if(view->getDimension() == 2){
        view->shape.insert(view->shape.begin(), 1);
        view->strides.insert(view->strides.begin(), 0); 
    }
    else {
        if(view->shape[0] == 1) { view->strides[0] = 0;}
    }
    return view;

}

TensorInfo Tensor::getInfo()  {
    return TensorInfo{
        data.get(),
        shape.data(),
        strides.data(),
        getDimension(),
        total_size
    };
}

// ====================
// View / Broadcasting
// ====================
vector<int> Tensor::broadcast_shapes(const vector<int>& shape_a, const vector<int>& shape_b) {
    int max_dims = std::max(shape_a.size(), shape_b.size());
    vector<int> out_shape(max_dims);
    
    int i_a = (int)shape_a.size() - 1;
    int i_b = (int)shape_b.size() - 1;
    int i_out = max_dims - 1;
    
    while (i_out >= 0) {
        int dim_a = (i_a >= 0) ? shape_a[i_a] : 1;
        int dim_b = (i_b >= 0) ? shape_b[i_b] : 1;
        
        if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
             // In a robust framework we would throw exception here
             out_shape[i_out] = std::max(dim_a, dim_b);
        } else {
             out_shape[i_out] = std::max(dim_a, dim_b);
        }
        
        i_a--; i_b--; i_out--;
    }
    return out_shape;
}

shared_ptr<Tensor> Tensor::broadcast_to(const vector<int>& target_shape) {
    if (this->shape == target_shape) return shared_from_this();

    auto view = make_shared<Tensor>(*this);
    view->shape = target_shape;
    view->strides.resize(target_shape.size());
    
    int offset = (int)target_shape.size() - (int)this->shape.size();
    
    for (int i = 0; i < target_shape.size(); ++i) {
        int original_idx = i - offset;
        
        if (original_idx < 0) {
            // New dimension (broadcasted) -> stride 0
            view->strides[i] = 0;
        } else {
            // Existing dimension
            if (this->shape[original_idx] == target_shape[i]) {
                view->strides[i] = this->strides[original_idx];
            } else {
                // Dimension 1 stretched -> stride 0
                view->strides[i] = 0;
            }
        }
    }
    return view;
}

shared_ptr<Tensor> Tensor::compute_binary_op(shared_ptr<Tensor> b, BinaryOp op){
    // 1. Calculate output shape
    vector<int> out_shape = broadcast_shapes(this->shape, b->shape);
    auto out = std::make_shared<Tensor>(out_shape);

    // 2. Create broadcasted views of inputs
    auto a_view = this->broadcast_to(out_shape);
    auto b_view = b->broadcast_to(out_shape);
    
    // 3. Update output info
    TensorInfo i_out = out->getInfo();
    
    // 4. Call backend with aligned shapes
    Device::get()->binary(a_view->getInfo(), b_view->getInfo(), i_out, op);

    return out;
}

shared_ptr<Tensor> Tensor::compute_unary_op(UnaryOp op) {
    auto out = std::make_shared<Tensor>(this->shape);
    TensorInfo i_out = out.get()->getInfo();
    
    Device::get()->unary(this->getInfo(), i_out, op);

    return out;
}

shared_ptr<Tensor> Tensor::compute_reduce_op(int dim, ReduceOp op) {
    vector<int> out_shape = this->shape;
    out_shape[dim] = 1;
    auto out = std::make_shared<Tensor>(out_shape);
    TensorInfo i_out = out.get()->getInfo();
    
    Device::get()->reduce(this->getInfo(), i_out, dim, op);

    return out;

}

// ===================
// Tensor initializers
// ===================
shared_ptr<Tensor> Tensor::zeros(const vector<int>& shape) {
    return make_shared<Tensor>(shape, vector<float>(element_vector_product(shape),0.0f));
}

shared_ptr<Tensor> Tensor::ones(const vector<int>& shape) {
    return make_shared<Tensor>(shape, vector<float>(element_vector_product(shape),1.0f));
}

shared_ptr<Tensor> Tensor::random(const vector<int>& shape, float min_val, float max_val) {
    size_t size = element_vector_product(shape);
    vector<float> random_data(size);
    
    static random_device rd; // Static so we don't need to reinializate the seed
    static mt19937 gen(rd());
    uniform_real_distribution<> dis(min_val, max_val);
    for (auto& val : random_data) val = dis(gen);

    return make_shared<Tensor>(shape, random_data);
}

// ===================
// Backward functions
// ===================

void Tensor::build_topo(shared_ptr<Tensor> v, vector<shared_ptr<Tensor>>& topo, unordered_set<Tensor*>& visited){
    if(visited.find(v.get()) == visited.end()){
        visited.insert(v.get());
        for(auto& child : v.get()->getParents()){
            build_topo(child, topo,visited);
        }
        topo.push_back(v);
    }

}

// backward function
void Tensor::backward() {
    vector<shared_ptr<Tensor>> topo;
    unordered_set<Tensor*> visited;
    
    // we build out topo graph
    build_topo(shared_from_this(), topo, visited);
    
    // we initialize the current gradient to ones
    this->grad = Tensor::ones(this->shape);
    
    // we go through the graph in reverse topological order
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        shared_ptr<Tensor> t = *it;
        // we apply the backward function
        if (t->_backward) {
            t->_backward();
        }
    }
}

// ===================
// Load and save weights
// ===================

void Tensor::serialize(std::ofstream& out) const {
    size_t dim = this->shape.size();
    out.write(reinterpret_cast<const char*>(&dim), sizeof(size_t));
    out.write(reinterpret_cast<const char*>(this->shape.data()), dim * sizeof(int));
    out.write(reinterpret_cast<const char*>(this->getData()), this->total_size * sizeof(float));

}

void Tensor::deserialize(std::ifstream& in)  {
    size_t dim;
    in.read(reinterpret_cast<char*>(&dim), sizeof(size_t));

    this->shape.resize(dim);
    in.read(reinterpret_cast<char*>(shape.data()), dim * sizeof(int));

    size_t total_elements = 1;
    for(int dim : this->shape) {
        total_elements *= dim;
    }

    this->data.reset(new float[total_elements]);
    in.read(reinterpret_cast<char*>(this->getData()), total_elements * sizeof(float));


}



