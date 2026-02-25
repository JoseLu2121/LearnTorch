#pragma once
#include <cstddef> 

// A tensor subclass with less information for maximizing performance
struct TensorInfo {
    float* data;
    const int* shape;
    const int* strides;
    int dim;
    size_t size;
};

// Enum definitions for operations
enum class BinaryOp {ADD, SUB, MUL, DIV, POW};
enum class UnaryOp {RELU, SIGMOID, TANH, EXP, LOG, NEG, SQRT};
enum class ReduceOp {SUM, MEAN, MAX, MIN, ARGMAX};
enum class JoinMode {SUM,CONCAT};