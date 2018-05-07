#pragma once

#include <algorithm>
#include <boost/multi_array.hpp>
#include <cmath>
#include <iostream>
#include <vector>

// ------------------------------------------------------------
#include "base/hash_specializations.hpp"
#include "laguerren_impl.hpp"

namespace boltzmann {

template <typename NUMERIC>
class LaguerreNKS
{
 public:
  typedef NUMERIC numeric_t;

 public:
  LaguerreNKS(int K)
      : Y_(K + 1)
      , K_(K)
      , is_initialized_(false)
  {
  }

  void compute(const std::vector<numeric_t> &x);

  /**
   *
   *
   * @param x
   * @param n
   * @param expw evaluation weight: e^(-r^2*expw)
   */
  void compute(const numeric_t *x, unsigned int n, numeric_t expw = numeric_t(0));

  unsigned int get_npoints() const { return Y_[0].shape()[1]; }

 public:
  typedef boost::multi_array<numeric_t, 2> array_t;

 public:
  const NUMERIC *get(unsigned int k, unsigned int alpha) const;
  void info() const;

 private:
  std::vector<array_t> Y_;
  unsigned int K_;
  bool is_initialized_;
};

// ----------------------------------------------------------------------
template <typename NUMERIC>
void
LaguerreNKS<NUMERIC>::compute(const std::vector<numeric_t> &x)
{
  compute(x.data(), x.size());
}

// ----------------------------------------------------------------------
template <typename NUMERIC>
void
LaguerreNKS<NUMERIC>::compute(const numeric_t *x, unsigned int n, numeric_t expw)
{
  std::vector<numeric_t> x2(n);
  std::transform(x, x + n, x2.begin(), [](const numeric_t &v) { return v * v; });
  // L_n-1
  std::vector<numeric_t> Lnm1(n);
  // L_n-2
  std::vector<numeric_t> Lnm2(n);

  for (unsigned int alpha = 0; alpha <= K_; ++alpha) {
    Y_[alpha].resize(boost::extents[K_ / 2 + 1][n]);
// init
#pragma omp parallel for
    for (size_t xi = 0; xi < n; ++xi) {
      numeric_t fexp = 1;
      if (::math::abs(expw) > 1e-16) fexp = ::math::exp(-expw * x2[xi]);
      Y_[alpha][0][xi] = boost::math::laguerren(0, alpha, x2[xi]) * fexp;
      Y_[alpha][1][xi] = boost::math::laguerren(1, alpha, x2[xi]) * fexp;
    }

    for (unsigned int kp = 2; kp <= K_ / 2; ++kp) {
#pragma omp parallel for
      for (size_t xi = 0; xi < n; ++xi) {
        Y_[alpha][kp][xi] = boost::math::laguerren_next(
            kp - 1, alpha, x2[xi], Y_[alpha][kp - 1][xi], Y_[alpha][kp - 2][xi]);
      }
    }
  }

  boost::multi_array<numeric_t, 2> powers_of_x(boost::extents[K_ + 1][n]);

  for (unsigned int xi = 0; xi < n; ++xi) {
    powers_of_x[0][xi] = 1.0;
  }

  for (unsigned int alpha = 1; alpha <= K_; ++alpha) {
    for (unsigned int xi = 0; xi < n; ++xi) {
      powers_of_x[alpha][xi] = x[xi] * powers_of_x[alpha - 1][xi];
    }
  }

  // add factor x^(2j+k%2)
  for (unsigned int alpha = 0; alpha <= K_; ++alpha) {
    for (unsigned int kp = 0; kp <= K_ / 2; ++kp) {
      int j = alpha / 2;
      int k = 2 * kp + 2 * j + alpha % 2;
      for (size_t xi = 0; xi < n; ++xi) {
        Y_[alpha][kp][xi] *= powers_of_x[2 * j + k % 2][xi];
      }
    }
  }

  is_initialized_ = true;
}

// ----------------------------------------------------------------------
template <typename NUMERIC>
const NUMERIC *
LaguerreNKS<NUMERIC>::get(unsigned int k, unsigned int alpha) const
{
  assert(is_initialized_);
  assert(alpha < Y_.size());
  assert(k < Y_[alpha].shape()[0]);
  return Y_[alpha][k].origin();
}

// ----------------------------------------------------------------------
template <typename NUMERIC>
void
LaguerreNKS<NUMERIC>::info() const
{
  unsigned long long int nentries = 0;
  for (unsigned int alpha = 0; alpha < Y_.size(); ++alpha) {
    nentries += Y_[alpha].shape()[0] * Y_[alpha].shape()[1];
  }

  std::cout << " LaguerreN uses " << nentries * sizeof(NUMERIC) / 1e6 << " MB" << std::endl;
}

}  // end namespace boltzmann
