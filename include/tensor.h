#pragma once
#include <vector>
#include <memory>
#include <functional>
#include <iostream>
#include <cassert>
#include <unordered_set>
#include <random>

// Tensor Class 
struct Tensor : std::enable_shared_from_this<Tensor> {
private:
    // Data storage 
    std::shared_ptr<float[]> data;
    // Number of elementes of the tensor
    size_t total_size;
    // Offset used for batches
    size_t offset = 0;

public:
    // Shape of the tensor (batch,row,col)
    std::vector<int> shape;
    // Strides for each dimension
    std::vector<int> strides;
    // Parent nodes of the tensor in the computation graph
    std::vector<std::shared_ptr<Tensor>> parents;
    
    // Gradient tensor
    std::shared_ptr<Tensor> grad;
    // Backward function
    std::function<void()> _backward;
    void backward();

    // Constructor
    Tensor(const std::vector<int>& shape_param, const std::vector<float>& data_param = {},
           const std::vector<std::shared_ptr<Tensor>> parents_param = {});
    
    // Copy constructor
    Tensor(const Tensor& other);

    // Prints the first n elements of the tensor
    void printElements(int count = 1) const;

    // Prints the shape of the tensor
    void printShape() const;

    //Prints the strides of the tensor
    void printStrides() const;

    // Prints all the info of the tensor with a max number of data elements
    void info(int max_size = 20) const;
    
    // Get the size of the tensor
    size_t getSize() const { return total_size; }

    // Get the vector of strides of the tensor
    std::vector<int> getStrides() const { return strides; }

    // Get the pointer to the data of the tensor
    float* getData() const { return data.get() + offset; }

    // Get the vector of the shape of the tensor
    std::vector<int> getShape() const { return shape; }

    // Get the parents of the tensor
    std::vector<std::shared_ptr<Tensor>> getParents() const { return parents; }
    int getDimension() const { return shape.size(); }
    
    // Get a batch given an index
    std::shared_ptr<Tensor> getBatch(int index);

    // Build a 3D view of a tensor
    std::shared_ptr<Tensor> view_to_3d(); 

    // Static methods to create special tensors
    static std::shared_ptr<Tensor> zeros(const std::vector<int>& shape); // All elements are 0

    static std::shared_ptr<Tensor> ones(const std::vector<int>& shape); // All elements are 1

    // Elements are random values between min_val and max_val
    static std::shared_ptr<Tensor> random(const std::vector<int>& shape, float min_val = -1.0f, float max_val = 1.0f);
    

    void build_topo(std::shared_ptr<Tensor> v,
                std::vector<std::shared_ptr<Tensor>>& topo,
                std::unordered_set<Tensor*>& visited);

};