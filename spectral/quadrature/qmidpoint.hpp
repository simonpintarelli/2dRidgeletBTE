#pragma once

#include "base/numbers.hpp"
#include "quadrature_base.hpp"

namespace boltzmann {

class QMidpoint : public Quadrature<1>
{
 public:
  typedef Quadrature<1> base_type;
  /**
   * Midpoint rule on [0, 2 Pi]
   *
   * @param npts
   *
   * @return
   */
  QMidpoint(int npts);
};
}
