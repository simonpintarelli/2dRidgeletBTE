#pragma once

#include <vector>

namespace boltzmann {

void
gauss_hermite_roots(std::vector<double> &roots, const int N, const int digits = 256);

}  // end namespace boltzmann
