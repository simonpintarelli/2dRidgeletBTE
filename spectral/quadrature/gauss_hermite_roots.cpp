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

namespace boltzmann {

namespace mp = boost::multiprecision;
using namespace mpfr;

typedef mp::mpfr_float_backend<100000> mfloat_t;
typedef mp::number<mfloat_t> mpfr_float_t;

using namespace std;

void gauss_hermite_roots(std::vector<double>& roots, const int N, const int ndigits)
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
  eigensolver.compute(A, Eigen::EigenvaluesOnly);

  const auto w = eigensolver.eigenvalues();

  for (int i = 0; i < N; ++i) {
    roots[i] = w(i).toDouble();
  }

  // sort
  std::sort(roots.begin(), roots.end());
}
}  // end namespace boltzmann
