#pragma once

#include "quadrature_base.hpp"


/**
 * @brief Gauss-Quadrature on [0,inf) with weight exp(-x*x)
 *        uses Golub-Welsch and a multiprecision eigenvalue solver
 *
 * @param N_
 * @param ndigits
 *
 * @return
 */
class GaussHermiteQuadrature : public Quadrature<1>
{
 private:
  typedef Quadrature<1> base_t;

 public:
  GaussHermiteQuadrature(int N_, int ndigits)
      : base_t(N_)
      , N(N_)
  {
    init(ndigits);
  }

  GaussHermiteQuadrature(int N)
      : GaussHermiteQuadrature(N, 256)
  {
  }

 protected:
  void init(int ndigits);

 private:
  using base_t::pts_;
  using base_t::wts_;
  int N;
};
