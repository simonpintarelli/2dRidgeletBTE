#pragma once

#include "maxwell_quadrature.hpp"


class QMaxwell : public Quadrature<1>
{
 private:
  typedef Quadrature<1> base_t;

 public:
  QMaxwell(double alpha, int N)
      : QMaxwell(alpha, N, 256)
  {
  }

  QMaxwell() { /* default constructor */}

  /**
   *
   *
   * @param alpha
   * @param N
   * @param ndigits  multiprecision for Golub-Welsch
   *
   * @return
   */
  QMaxwell(double alpha, int N, int ndigits);

 private:
  using base_t::pts_;
  using base_t::wts_;
};

inline QMaxwell::QMaxwell(double alpha, int N, int ndigits)
    : base_t(N)
{
  MaxwellQuadrature qbase(N, ndigits);

  for (int i = 0; i < N; ++i) {
    pts_[i] = qbase.pts(i) / std::sqrt(alpha);
    wts_[i] = qbase.wts(i) / alpha;
  }
}
