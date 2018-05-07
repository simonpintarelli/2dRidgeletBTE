// system includes ------------------------------------------------------------
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/MPRealSupport>
#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include <iomanip>
#include <iostream>
//#include <quadmath.h>

// own includes ---------------------------------------------------------------
#include "base/numbers.hpp"
#include "maxwell_quadrature.hpp"
#include "spectral/mpfr/import_std_math.hpp"

namespace mp = boost::multiprecision;  // Reduce the typing a bit later...

using namespace mpfr;

typedef mp::mpfr_float_backend<100000> mfloat_t;
typedef mp::number<mfloat_t> mpfr_float_t;

using namespace std;

const mpfr_float_t PI = boost::math::constants::pi<mpfr_float_t>();
const mpfr_float_t rPI = boost::math::constants::root_pi<mpfr_float_t>();

template <typename T>
class Beta
{
 public:
  Beta(int p = 1)
      : p_(p)
      , beta_n(1)
  {
    beta_n[0] = 0;
    navail = 0;
  }

  template <typename A>
  const T& get(unsigned int n, A& alpha);

 private:
  int p_;
  std::vector<T> beta_n;
  unsigned int navail;
};

// ----------------------------------------------------------------------
template <typename T>
template <typename A>
const T& Beta<T>::get(unsigned int n, A& alpha)
{
  if (n < beta_n.size())
    return beta_n[n];
  else {
    // make sure all previous n's exist
    for (unsigned int i = navail + 1; i < n; ++i) {
      this->get(i, alpha);
    }
    // compute
    T am = alpha.get(n - 1, *this);
    T an = alpha.get(n, *this);
    T bm = this->get(n - 1, alpha);
    T denom = 2 + 4 * am * (am + an) + 4 * bm;
    T nom = n * (n + p_);
    T res = nom / denom;
    beta_n.push_back(res);
    navail++;
    assert(navail == n);
    assert(beta_n[n] == res);
    return beta_n[n];
  }
}

// ----------------------------------------------------------------------
template <typename T>
class Alpha
{
 public:
  Alpha(int p = 1)
      : p_(p)
      , alpha_n(2)
  {
    mpfr_float_t one(1);
    mpfr_float_t two(2);
    mpfr_float_t three(3);
    mpfr_float_t four(4);
    mpfr_float_t eight(8);

    if (p == 0) {
      alpha_n[0] = one / rPI;
      alpha_n[1] = two / rPI / (PI - two);
    } else if (p == 1) {
      alpha_n[0] = rPI / two;
      alpha_n[1] = rPI * (PI - two) / two / (four - PI);
    } else if (p == 2) {
      alpha_n[0] = two / rPI;
      alpha_n[1] = four * (four - PI) / rPI / (three * PI - eight);
    } else {
      throw std::runtime_error("invalid p");
    }

    navail = 1;
  }
  template <typename B>
  const T& get(unsigned int n, B& beta);

 private:
  int p_;
  std::vector<T> alpha_n;
  unsigned int navail;
};

// ----------------------------------------------------------------------
template <typename T>
template <typename B>
const T& Alpha<T>::get(unsigned int n, B& beta)
{
  if (n < alpha_n.size())
    return alpha_n[n];
  else {
    // make sure all previous n's exist
    for (unsigned int i = navail + 1; i < n; ++i) {
      this->get(i, beta);
    }
    // use recursion formula (15)
    // compute
    T sum = 0;
    T am = this->get(n - 1, beta);
    T am2 = am * am;
    for (unsigned int k = 0; k < n; ++k) {
      sum += this->get(k, beta);
    }
    sum /= (2 * n - 1 + p_ - 2 * am2 - 2 * beta.get(n - 1, *this));
    sum -= am;
    alpha_n.push_back(sum);
    navail++;
    assert(navail == n);
    assert(alpha_n[n] == sum);
    return alpha_n[n];
  }
}

void MaxwellQuadrature::init(int ndigits)
{
  // recurrence relation using mpfr_float_t
  typedef Alpha<mpfr_float_t> alpha_t;
  typedef Beta<mpfr_float_t> beta_t;

  alpha_t alpha(p);
  beta_t beta(p);

  // eigen with mpreal
  mpreal::set_default_prec(ndigits);
  typedef mpreal mfloat_t;
  //  typedef double mfloat_t;
  typedef Eigen::Matrix<mfloat_t, Eigen::Dynamic, Eigen::Dynamic> MatrixXmp;
  MatrixXmp A(N, N);
  A.fill(0);
  for (int i = 0; i < N; ++i) {
    A(i, i) = mfloat_t(alpha.get(i, beta).backend().data()).setPrecision(ndigits);
    const auto sqrt_beta1 = ::math::sqrt(beta.get(i + 1, alpha));
    const auto sqrt_beta0 = ::math::sqrt(beta.get(i, alpha));

    if (i + 1 < N) A(i, i + 1) = mfloat_t(sqrt_beta1.backend().data()).setPrecision(ndigits);
    if (i > 0) A(i, i - 1) = mfloat_t(sqrt_beta0.backend().data()).setPrecision(ndigits);
  }

  Eigen::SelfAdjointEigenSolver<MatrixXmp> eigensolver;
  eigensolver.compute(A, Eigen::ComputeEigenvectors);

  // compute weights
  const auto V = eigensolver.eigenvectors();
  const auto w = eigensolver.eigenvalues();

  if (V.rows() == 0) {
    cerr << "abort";
    exit(-1);
  }

  const double pi = boltzmann::numbers::PI;
  std::array<double, 3> mp = {std::sqrt(pi) / 2, 0.5, std::sqrt(pi) / 4};

  std::vector<double> weights(N);
  for (int i = 0; i < N; ++i) {
    // eigenvectors are normalized by eigen
    auto v = V(0, i);
    weights[i] = mp[p] * (v * v).toDouble();
  }

  std::vector<double> points(N);
  for (int i = 0; i < N; ++i) {
    points[i] = w(i).toDouble();
  }

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
