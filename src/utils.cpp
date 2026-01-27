#include "utils.h"
#include <numeric>

size_t element_vector_product(const std::vector<int>& v) {
    if (v.empty()) return 0;
    size_t total = 1;
    for (int dim : v) total *= dim;
    return total;
}

