#include "tensor.h"
#include "neuron.h"
#include "ops.h"
#include "utils.h" 
using namespace std;


Neuron::Neuron(int input_size, Activation activation_function) : input_size(input_size) {
    uniform_real_distribution<> dist(-1.0, 1.0);
    this->weights = Tensor::random({input_size, 1});
    this->bias = Tensor::ones({1});
    this->input_size = input_size;

    if(activation_function) {
        this->activation_function = activation_function;
    } else {
        this->activation_function = linear_activation;
    }
}


std::shared_ptr<Tensor> Neuron::forward(const std::shared_ptr<Tensor>& input) {

    auto w_t = transpose_view(weights);

    auto z = matmul(input, w_t) + bias; 
    output = activation_function(z); 
    return output;
}






