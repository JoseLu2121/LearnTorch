#include "utils.h"
#include <numeric>

size_t element_vector_product(const std::vector<int>& v) {
    return std::accumulate(v.begin(), v.end(), 1, std::multiplies<int>());
}