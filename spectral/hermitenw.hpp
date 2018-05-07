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
class HermiteNW : public PolyBase<T>
{
 public:
  using typename PolyBase<T>::numeric_t;

 public:
  HermiteNW(unsigned int n);
  void compute(const std::vector<numeric_t> &x);

 private:
  using PolyBase<T>::Y_;
  using PolyBase<T>::n_;
};

// ----------------------------------------------------------------------
template <typename T>
HermiteNW<T>::HermiteNW(unsigned int n)
    : PolyBase<T>(n)
{ /* empty */
}

// ----------------------------------------------------------------------
template <typename T>
void
HermiteNW<T>::compute(const std::vector<numeric_t> &x)
{
  Y_.resize(boost::extents[n_ + 1][x.size()]);
  unsigned int N = x.size();

  std::vector<numeric_t> expw(x.size());
  for (unsigned int i = 0; i < N; ++i) {
    expw[i] = ::math::exp(-x[i] * x[i] * 1 / numeric_t(2));
  }

  // initalize l = 0
  for (unsigned int i = 0; i < N; ++i) {
    Y_[0][i] = boost::math::hermiten(0, x[i]) * expw[i];
    Y_[1][i] = boost::math::hermiten(1, x[i]) * expw[i];
  }

  for (unsigned int l = 1; l < n_; ++l) {
    //#pragma omp parallel for
    for (unsigned int i = 0; i < N; ++i) {
      Y_[l + 1][i] = boost::math::hermiten_next(l, x[i], Y_[l][i], Y_[l - 1][i]);
    }
  }
}

}  // end namespace boltzmann
