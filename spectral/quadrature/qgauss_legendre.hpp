#pragma once

#include "gauss_legendre_quadrature.hpp"
#include "quadrature_base.hpp"


class QGaussLegendre : public Quadrature<1>
{
 private:
  typedef Quadrature<1> base_t;

 public:
  QGaussLegendre(const unsigned int N, const double a = -1, const double b = 1);

 private:
  using base_t::pts_;
  using base_t::wts_;
};

// ----------------------------------------------------------------------
QGaussLegendre::QGaussLegendre(const unsigned int N, const double a, const double b)
    : Quadrature<1>(N)
{
  const double w = b - a;
  assert(b > a);

  GaussLegendreQuadrature quad(N);

  for (unsigned int i = 0; i < N; ++i) {
    base_t::pts_[i] = w * (quad.pts(i) + 1.0) / 2.0 + a;
    base_t::wts_[i] = quad.wts(i) / 2.0 * w;
  }
}
