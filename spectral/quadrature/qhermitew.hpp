#pragma once

#include <cmath>

#include "gauss_hermite_roots.hpp"
#include "quadrature_base.hpp"
#include "spectral/hermiten_impl.hpp"


namespace boltzmann {

/**
 * \f$ \int_\mathbb{R} g(x) exp(-x^2) \mathrm{d}\,x \approx \sum_{i=1}^n (g(x_i)
 * exp(-x^2)) w_i \f$
 *
 * \remark{In order to avoid numerical  underflow in the weights \f$ w_i \f$ and
 * overflow in
 * the evaluation of the integrad, the weights contain the factor \f$ exp(x_i^2)
 * \f$}
 *
 * @param alpha scaling parameter \f$ exp(-\alpha x^2)\f$
 * @param N no. of quadrature points
 *
 * @return
 */
class QHermiteW : public Quadrature<1>
{
 private:
  typedef Quadrature<1> base_t;

 public:
  QHermiteW(double alpha, int N)
      : QHermiteW(alpha, N, 256)
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
  QHermiteW(double alpha, int N, int ndigits);
  QHermiteW() {}

 private:
  using base_t::pts_;
  using base_t::wts_;
};

inline QHermiteW::QHermiteW(double alpha, int N, int ndigits)
    : base_t(N)
{
  gauss_hermite_roots(pts_, N, ndigits);

  const double ah2 = std::sqrt(alpha);
  // ------------------------------------------------------------
  // compute weights by formula
  std::vector<double> hn(N);
  std::vector<double> hnm(N);
  for (int i = 0; i < N; ++i) {
    const double expw = std::exp(-0.5 * pts_[i] * pts_[i]);
    hnm[i] = boost::math::hermiten(0, pts_[i]) * expw;
    hn[i] = boost::math::hermiten(1, pts_[i]) * expw;
  }
  for (int n = 2; n <= N - 1; ++n) {
    for (int i = 0; i < N; ++i) {
      hnm[i] = boost::math::hermiten_next(n - 1, pts_[i], hn[i], hnm[i]);
    }
    std::swap(hn, hnm);
  }
  // compute weights and scale nodes
  for (int i = 0; i < N; ++i) {
    wts_[i] = 1.0 / (N * hn[i] * hn[i]) / ah2;
    pts_[i] /= ah2;
  }
}

}  // end namespace boltzmann
