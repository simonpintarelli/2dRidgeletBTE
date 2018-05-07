#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/MPRealSupport>
#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include <iomanip>
#include <iostream>

#include "gauss_hermite_quadrature.hpp"
#include "spectral/mpfr/import_std_math.hpp"

//#include <quadmath.h>

namespace mp = boost::multiprecision;  // Reduce the typing a bit later...

using namespace mpfr;

typedef mp::mpfr_float_backend<100000> mfloat_t;
typedef mp::number<mfloat_t> mpfr_float_t;

using namespace std;

void GaussHermiteQuadrature::init(int ndigits)
{
  mpreal::set_default_prec(ndigits);
  typedef mpreal mfloat_t;
  //  typedef double mfloat_t;
  typedef Eigen::Matrix<mfloat_t, Eigen::Dynamic, Eigen::Dynamic> MatrixXmp;
  MatrixXmp A(N, N);
  A.fill(0);
  for (int i = 0; i < N - 1; ++i) {
    const double beta = ::math::sqrt(1.0 * (i + 1)) / ::math::sqrt(2.0);
    A(i, i + 1) = beta;
    A(i + 1, i) = beta;
  }

  Eigen::SelfAdjointEigenSolver<MatrixXmp> eigensolver;
  // Eigen::EigenSolver<MatrixXmp> eigensolver;
  eigensolver.compute(A, Eigen::ComputeEigenvectors);
  const double pi = boost::math::constants::pi<double>();
  const double m = ::math::sqrt(pi);
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
