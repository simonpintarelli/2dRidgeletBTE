#pragma once

#include <boost/math/special_functions/hermite.hpp>

#include "hermiten_impl.hpp"
#include "poly_base.hpp"

namespace boltzmann {

/**
 * @brief Physicists' Hermite Polynomials normalized
 *
 */
template <typename T>
class HermiteN : public PolyBase<T>
{
 private:
  using typename PolyBase<T>::numeric_t;

 public:
  HermiteN(unsigned int n);
  void compute(const std::vector<numeric_t> &x);

 private:
  using PolyBase<T>::Y_;
  using PolyBase<T>::n_;
  using PolyBase<T>::is_initialized_;
};

// ----------------------------------------------------------------------
template <typename T>
HermiteN<T>::HermiteN(unsigned int n)
    : PolyBase<T>(n)
{ /* empty */
}

// ----------------------------------------------------------------------
template <typename T>
void
HermiteN<T>::compute(const std::vector<numeric_t> &x)
{
  Y_.resize(boost::extents[n_ + 1][x.size()]);
  unsigned int N = x.size();
  // initalize l = 0
  for (unsigned int i = 0; i < N; ++i) {
    Y_[0][i] = boost::math::hermiten(0, x[i]);
    if (n_ > 0) Y_[1][i] = boost::math::hermiten(1, x[i]);
  }

  for (unsigned int l = 1; l < n_; ++l) {
    //#pragma omp parallel for
    for (unsigned int i = 0; i < N; ++i) {
      Y_[l + 1][i] = boost::math::hermiten_next(l, x[i], Y_[l][i], Y_[l - 1][i]);
    }
  }

  is_initialized_ = true;
}

}  // end namespace boltzmann
