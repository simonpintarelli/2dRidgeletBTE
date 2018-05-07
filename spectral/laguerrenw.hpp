#pragma once

#include <boost/multi_array.hpp>
#include <cmath>
#include <iostream>
#include <vector>

// ------------------------------------------------------------
#include "base/hash_specializations.hpp"
#include "laguerren_impl.hpp"

template <typename NUMERIC>
class LaguerreNW
{
 public:
  typedef NUMERIC numeric_t;

 public:
  LaguerreNW(int K)
      : Y_(K + 1)
      , K_(K)
  {
  }

  void compute(const std::vector<numeric_t> &x);
  void compute(const numeric_t *x, unsigned int n);

  unsigned int get_npoints() const { return Y_[0].shape()[1]; }

 public:
  typedef boost::multi_array<numeric_t, 2> array_t;

 public:
  const NUMERIC *get(unsigned int k, unsigned int alpha) const;
  void info() const;

 private:
  std::vector<array_t> Y_;
  unsigned int K_;
};

// ----------------------------------------------------------------------
template <typename NUMERIC>
void
LaguerreNW<NUMERIC>::compute(const std::vector<numeric_t> &x)
{
  compute(x.data(), x.size());
}

// ----------------------------------------------------------------------
template <typename NUMERIC>
void
LaguerreNW<NUMERIC>::compute(const numeric_t *x, unsigned int n)
{
  // L_n-1
  std::vector<numeric_t> Lnm1(n);
  // L_n-2
  std::vector<numeric_t> Lnm2(n);

  for (unsigned int alpha = 0; alpha <= K_; ++alpha) {
    Y_[alpha].resize(boost::extents[K_ / 2 + 1][n]);
// init
#pragma omp parallel for
    for (size_t xi = 0; xi < n; ++xi) {
      numeric_t expw = ::math::exp(-0.5 * x[xi]);
      Y_[alpha][0][xi] = boost::math::laguerren(0, alpha, x[xi]) * expw;
      Y_[alpha][1][xi] = boost::math::laguerren(1, alpha, x[xi]) * expw;
    }

    for (unsigned int k = 2; k <= K_ / 2; ++k) {
#pragma omp parallel for
      for (size_t xi = 0; xi < n; ++xi) {
        Y_[alpha][k][xi] = boost::math::laguerren_next(
            k - 1, alpha, x[xi], Y_[alpha][k - 1][xi], Y_[alpha][k - 2][xi]);
      }
    }
  }
}

// ----------------------------------------------------------------------
template <typename NUMERIC>
const NUMERIC *
LaguerreNW<NUMERIC>::get(unsigned int k, unsigned int alpha) const
{
  assert(alpha < Y_.size());
  assert(k < Y_[alpha].shape()[0]);
  return Y_[alpha][k].origin();
}

// ----------------------------------------------------------------------
template <typename NUMERIC>
void
LaguerreNW<NUMERIC>::info() const
{
  unsigned long long int nentries = 0;
  for (unsigned int alpha = 0; alpha < Y_.size(); ++alpha) {
    nentries += Y_[alpha].shape()[0] * Y_[alpha].shape()[1];
  }

  std::cout << " LaguerreNW uses " << nentries * sizeof(NUMERIC) / 1e6 << " MB" << std::endl;
}
