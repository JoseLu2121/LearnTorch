#include "tensor.h"
#include "utils.h" 
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
      offset(other.offset),
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
        for (int i = 0; i < max_size; i++) cout << data[i + offset] << " ";
        cout << "]" << endl;
    } else {
        cout << "  Data: [ ... " << total_size << " elementos ... ]" << endl;
    }
    cout << "-------------------------" << endl;
}

// ===================
// Size and view manipulation
// ===================

shared_ptr<Tensor> Tensor::getBatch(int index) {

    // We can only get a batch of a 3D tensor
    if(getDimension() != 3) throw std::runtime_error("getBatch only support 3D tensors");

    // We must throw an error if the index is out of range
    if (index < 0 || index >= shape[0])
        throw std::out_of_range("Batch index out of range");

    // We create a view of the tensor
    auto tensor_view = make_shared<Tensor>(*this);

    tensor_view->shape = {shape[1], shape[2]};
    tensor_view->offset = this->offset + (index * strides[0]);
    tensor_view->strides = {strides[1], strides[2]};
    tensor_view->parents = {shared_from_this()}; // The parent of this tensor is the original one

    return tensor_view;
}

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

