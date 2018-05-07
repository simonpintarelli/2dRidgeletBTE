#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

// own includes ----------------------------------------
#include "gauss_laguerre.hpp"
#include "quadrature_base.hpp"


namespace boltzmann {

class QGaussLaguerre : public Quadrature<1>
{
 public:
  typedef Quadrature<1> base_type;
  QGaussLaguerre(double alpha, int npoints)
      : base_type(npoints)
  {
    assert(alpha > 0);
    LaguerreQuad quad(npoints);
    const auto &rwts = quad.get_weights();
    const auto &rpts = quad.get_points();
    std::transform(
        rpts.begin(), rpts.end(), pts_.begin(), [&](double r) { return sqrt(r / alpha); });
    std::transform(
        rwts.begin(), rwts.end(), wts_.begin(), [&](double w) { return 1. / (2 * alpha) * w; });
  }

  const decltype(pts_) &get_points() const { return pts_; }
  const decltype(wts_) &get_weights() const { return wts_; }
};
}  // end namespace boltzmann
