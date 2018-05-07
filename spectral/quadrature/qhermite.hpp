#pragma once

#include <cmath>

#include <spectral/hermiten_impl.hpp>
#include "gauss_hermite_quadrature.hpp"


namespace boltzmann {

class QHermite : public Quadrature<1>
{
 private:
  typedef Quadrature<1> base_t;

 public:
  QHermite(double alpha, int N)
      : QHermite(alpha, N, 128)
  {
  }

  /**
   *
   *
   * @param alpha
   * @param N
   * @param ndigits #digits for mpfr
   *
   * @return
   */
  QHermite(double alpha, int N, int ndigits);

 private:
  using base_t::pts_;
  using base_t::wts_;
};

inline QHermite::QHermite(double alpha, int N, int ndigits)
    : base_t(N)
{
  GaussHermiteQuadrature qbase(N, ndigits);

  for (int i = 0; i < N; ++i) {
    pts_[i] = qbase.pts(i) * std::sqrt(alpha);
    wts_[i] = qbase.wts(i) * std::sqrt(alpha);
  }
}

}  // end namespace boltzmann
