#pragma once
#include <vector>
#include <cstddef>
#include "tensor.h"

size_t element_vector_product(const std::vector<int>& v);


inline bool is_contiguous(const TensorInfo& tensor) {
    int expected_stride = 1;
    for (int d = tensor.dim - 1; d >= 0; --d) {
        if (tensor.shape[d] != 1) { 
            if (tensor.strides[d] != expected_stride) {
                return false;
            }
            expected_stride *= tensor.shape[d];
        }
    }
    return true;
}