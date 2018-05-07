#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <array>
#include <boost/lexical_cast.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/multi_array.hpp>
#include <vector>

#include <cmath>
#include <fstream>
#include <iostream>

#include "quadrature/qhermitew.hpp"
#include "spectral/hermitenw.hpp"
#include "spectral/lagrange_polynomial.hpp"
#include "spectral/mpfr/import_std_math.hpp"

namespace boltzmann {

/**
 * Nodal basis: Lagrange polynomials at Hermite quadrature nodes
 * For Lagrange polynomial it holds
 * \f$ l_i(x_j) = 0 i \neq j \f$
 * \f$ l_i(x_i) = 1/sqrt(w_i) \f$
 *
 *
 * @param K
 */
template <typename NUMERIC_T = double>
class H2N_1d
{
 public:
  typedef NUMERIC_T numeric_t;
  //  typedef Eigen::Matrix<numeric_t, Eigen::Dynamic, Eigen::Dynamic> matrix_t;

 public:
  // H2N_1d() {}
  // H2N_1d(int K);

  /**
   *
   *
   * @param H2N  hermite to nodal matrix
   * @param N2H  nodal to hermite matrix
   * @param K    max. polynomial degree
   */
  template <typename MATRIX>
  static void create(MATRIX &H2N, MATRIX &N2H, const int K);
  //  const matrix_t& get_matrix() const { return N2H_; }

  // private:
  //   matrix_t N2H_;
};

// //
// --------------------------------------------------------------------------------
// template <typename NUMERIC_T>
// H2N_1d<NUMERIC_T>::
// H2N_1d(int K)
// { this->init(K); }

// --------------------------------------------------------------------------------
template <typename NUMERIC_T>
template <typename MATRIX>
void
H2N_1d<NUMERIC_T>::create(MATRIX &H2N, MATRIX &N2H, const int K)
{
  N2H.resize(K, K);
  H2N.resize(K, K);
  //  N2H_.resize(K,K);
  QHermiteW quad(1, K);
  HermiteNW<numeric_t> hermw(K);
  hermw.compute(quad.pts());

  for (int i = 0; i < K; ++i) {
    for (int j = 0; j < K; ++j) {
      const double wi = quad.wts(i);
      H2N(i, j) = hermw.get(j)[i] * ::math::sqrt(wi);
    }
  }

  N2H = H2N.transpose();
}

// -------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------

/**
 * @brief same as H2N_1d, but with nodes according to Gauss-Hermite quad with
 * weight
 * \f$ exp(-\alpha r^2) \f$
 *
 * ATTENTION: H2N(a=1) is orthonormal, but not for a != 1 !!!
 *
 * If this is used in conjunction with Hermite2Nodal, the `N->H` will give wrong
 * results! Only
 * use H->N.
 *  TODO: update code such that BOTH forward AND BACKWARD transformation works.
 *
 */
class H2NG_1d
{
 public:
  template <typename MATRIX>
  static void create(MATRIX &H2N, MATRIX &N2H, const int K, const double alpha)
  {
    typedef Eigen::VectorXd vec_t;

    H2N.resize(K, K);

    QHermiteW quad4nodes(alpha, K);
    QHermiteW quad(1, K);

    // Lagrange Poly
    vec_t xi = Eigen::Map<const vec_t>(quad4nodes.points_data(), K);
    vec_t li(K);  // lambda's (barycentric interpolation)
    vec_t yi(K);
    vec_t x = Eigen::Map<const vec_t>(quad.points_data(), K);
    vec_t w = Eigen::Map<const vec_t>(quad.weights_data(), K);
    vec_t y(K);
    // prepare barycentric interpol. weights
    LagrangePolynomial<>::compute_weights(li, xi);
    // evaluate hermite polynomials at quadrature points
    HermiteNW<double> hermw(K);
    hermw.compute(quad.pts());
    const double ah4 = std::pow(alpha, 0.25);

    Eigen::MatrixXd T(K, K);
    for (int i = 0; i < K; ++i) {
      yi.setZero();
      yi[i] = 1.0;
      LagrangePolynomial<>::evaluate(y, x, xi, yi, li);
      const double sqrtwi = std::sqrt(w[i]);
      for (int j = 0; j < K; ++j) {
        double sum = 0;
        // quadrature
        for (int q = 0; q < K; ++q) {
          sum += hermw.get(j)[q] * std::exp((x[i] * x[i] - x[q] * x[q]) / 2) * ah4 * y[q] * w[q] /
                 sqrtwi;
        }
        T(i, j) = sum;
      }
    }

    Eigen::MatrixXd M(K, K);

    vec_t y1(K);
    vec_t y2(K);

    const double ah2 = std::sqrt(alpha);
    // mass matrix (cf. notes in yellow notebook)
    for (int i = 0; i < K; ++i) {
      yi.setZero();
      yi[i] = 1.0;
      LagrangePolynomial<>::evaluate(y1, x, xi, yi, li);

      for (int j = 0; j < K; ++j) {
        yi.setZero();
        yi[j] = 1.0;
        LagrangePolynomial<>::evaluate(y2, x, xi, yi, li);

        double sum = 0;
        for (int q = 0; q < K; ++q) {
          sum += std::exp((x[i] * x[i] + x[j] * x[j]) / 2 - x[q] * x[q]) * y1[q] * y2[q] /
                 std::sqrt(w[i] * w[j]) * w[q];
        }
        M(i, j) = ah2 * sum;
      }
    }

    H2N = M.inverse() * T;
    N2H = T.transpose();
  }
};

}  // end namespace boltzmann
