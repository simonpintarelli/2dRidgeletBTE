#include "gauss_legendre_quadrature.hpp"
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/MPRealSupport>
#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include <iomanip>
#include <iostream>

//#include <quadmath.h>

namespace mp = boost::multiprecision;  // Reduce the typing a bit later...

using namespace mpfr;

typedef mp::mpfr_float_backend<100000> mfloat_t;
typedef mp::number<mfloat_t> mpfr_float_t;

using namespace std;

void GaussLegendreQuadrature::init(int ndigits)
{
  mpreal::set_default_prec(ndigits);
  typedef mpreal mfloat_t;
  //  typedef double mfloat_t;
  typedef Eigen::Matrix<mfloat_t, Eigen::Dynamic, Eigen::Dynamic> MatrixXmp;
  MatrixXmp A(N, N);
  A.fill(0);
  for (int i = 0; i < N - 1; ++i) {
    const double beta = (i + 1) / std::sqrt(4.0 * std::pow(i + 1, 2.0) - 1);
    A(i, i + 1) = beta;
    A(i + 1, i) = beta;
  }

  Eigen::SelfAdjointEigenSolver<MatrixXmp> eigensolver;
  // Eigen::EigenSolver<MatrixXmp> eigensolver;
  eigensolver.compute(A, Eigen::ComputeEigenvectors);

  const double m = 2.0;
  // compute weights
  const auto V = eigensolver.eigenvectors();

  if (V.rows() == 0) {
    cerr << "abort";
    exit(-1);
  }
  std::vector<double> weights(N);
  for (int i = 0; i < N; ++i) {
    // eigenvectors are normalized by eigen
    weights[i] = (V(0, i) * V(0, i) * m).toDouble();
  }

  const auto w = eigensolver.eigenvalues();
  std::vector<double> points(N);
  for (int i = 0; i < N; ++i) {
    points[i] = w(i).toDouble();
  }

  // sort
  std::vector<std::pair<double, double> > pairs(N);
  for (int i = 0; i < N; ++i) {
    pairs[i] = std::make_pair(points[i], weights[i]);
  }

  std::sort(pairs.begin(), pairs.end());

  for (int i = 0; i < N; ++i) {
    pts_[i] = pairs[i].first;
    wts_[i] = pairs[i].second;
  }
}
