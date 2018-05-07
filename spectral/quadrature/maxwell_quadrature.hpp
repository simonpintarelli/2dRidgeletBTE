#pragma once

#include "quadrature_base.hpp"


/**
 * @brief Gauss-Quadrature on [0,inf) with weight x exp(-x*x)
 *
 * Reference:
 *  B. Shizgal, A Gaussian quadrature procedure for use in the solution of the
 *  Boltzmann equation and related problems
 *  doi: 10.1016/0021-9991(81)90099-1
 *
 * @param N_
 * @param ndigits
 *
 * @return
 */
class MaxwellQuadrature : public Quadrature<1>
{
 private:
  typedef Quadrature<1> base_t;

 public:
  MaxwellQuadrature(int N_, int ndigits, int p_ = 1)
      : base_t(N_)
      , N(N_)
      , p(p_)
  {
    init(ndigits);
  }

  MaxwellQuadrature(int N)
      : MaxwellQuadrature(N, 256, 1)
  {
  }

 protected:
  void init(int ndigits);

 private:
  using base_t::pts_;
  using base_t::wts_;
  int N;
  int p;
};
