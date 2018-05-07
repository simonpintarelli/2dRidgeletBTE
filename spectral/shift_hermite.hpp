#pragma once

#include <boost/mpl/identity.hpp>
#include <Eigen/Dense>
#include <algorithm>
#include <boost/multi_array.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include <fstream>
#include <iomanip>
#include <stdexcept>

#include "spectral/hermiten.hpp"
#include "spectral/mpfr/import_std_math.hpp"

namespace boltzmann {

// ----------------------------------------------------------------------
namespace detail {
template <typename NUMERIC_T>
class sentry
{
 public:
  typedef NUMERIC_T numeric_t;

 private:
  typedef std::vector<numeric_t> vec_t;

 public:
  sentry(int n) { this->init(n); }

  sentry() { /* empty */}

  void init(int n)
  {
    n_ = n;
    factors_.resize(6);
    std::for_each(factors_.begin(), factors_.end(), [&](vec_t& v) { v.reserve(n + 1); });
  }

  NUMERIC_T coeff(int i, int j, int t);
  NUMERIC_T operator()(int i, int j, numeric_t x);

 private:
  int n_;
  std::vector<vec_t> factors_;
};

// ----------------------------------------------------------------------
template <typename NUMERIC_T>
NUMERIC_T
sentry<NUMERIC_T>::coeff(int i, int j, int t)
{
  const int maxij = std::max(i, j);
  const int minij = std::min(i, j);
  // (min(i,j) ... 1)
  auto& v0 = factors_[0];
  v0.resize(minij);
  for (int k = 0; k < minij; ++k) {
    v0[k] = std::min(i, j) - k;
  }

  //  sqrt(max(i,j) .. min(i,j)+1)
  auto& v1 = factors_[1];
  v1.resize(maxij - minij);
  for (int k = 0; k < maxij - minij; ++k) {
    v1[k] = ::math::sqrt(numeric_t(maxij - k));
  }

  // (i-t)!
  auto& v2 = factors_[2];
  v2.resize(i - t);
  for (int k = 0; k < i - t; ++k) {
    v2[k] = 1 / numeric_t(i - t - k);
  }

  // (j-t)!
  auto& v3 = factors_[3];
  v3.resize(j - t);
  for (int k = 0; k < j - t; ++k) {
    v3[k] = 1 / numeric_t(j - t - k);
  }

  // t!
  auto& v4 = factors_[4];
  v4.resize(t);
  for (int k = 0; k < t; ++k) {
    v4[k] = 1 / numeric_t(t - k);
  }

  // 2^(t-(i+j)/2)
  auto& v5 = factors_[5];
  int exp2 = t - (i + j) / 2;
  v5.resize(std::abs(exp2));
  if (exp2 > 0)
    for (unsigned int k = 0; k < v5.size(); ++k) {
      v5[k] = 2;
    }
  else
    for (unsigned int k = 0; k < v5.size(); ++k) {
      v5[k] = 1 / numeric_t(2);
    }
  // cout << "v5:\t";
  // for_each(v5.begin(), v5.end(), [](numeric_t v) { cout << v << "\t"; });
  // cout << endl;

  std::sort(factors_.begin(), factors_.end(), [](const vec_t& v1, const vec_t& v2) {
    return v1.size() < v2.size();
  });
  std::vector<int> lengths;
  std::for_each(
      factors_.begin(), factors_.end(), [&](const vec_t& v) { lengths.push_back(v.size()); });

  numeric_t f = (std::abs(i - t) % 2) ? -1 : 1;
  // multiply
  for (int jp = 0; jp < lengths[0]; ++jp) {
    numeric_t loc = 1;
    for (unsigned int fi = 0; fi < factors_.size(); ++fi) {
      loc *= factors_[fi][jp];
    }
    f *= loc;
  }
  for (unsigned int l = 1; l < lengths.size(); ++l) {
    for (int jp = lengths[l - 1]; jp < lengths[l]; ++jp) {  // loop over remaining positions
      numeric_t loc = 1;
      for (unsigned int fi = l; fi < factors_.size(); ++fi) {
        loc *= factors_[fi][jp];
      }
      f *= loc;
    }
  }

  // 2^(t-(i+j)/2) missing term
  if ((i + j) % 2 && exp2 <= 0)
    f /= ::math::sqrt(numeric_t(2));
  else if ((i + j) % 2 && exp2 > 0)
    f *= ::math::sqrt(numeric_t(2));

  return f;
}

// ----------------------------------------------------------------------
template <typename NUMERIC_T>
NUMERIC_T
sentry<NUMERIC_T>::operator()(int i, int j, numeric_t x)
{
  numeric_t sum = 0;

  // TOOD implement Horner's scheme
  for (int l = std::abs(i - j); l <= i + j; l += 2) {
    numeric_t loc = this->coeff(i, j, (i + j - l) / 2);

    sum += loc * ::math::pow(x, l);
  }

  return sum * ::math::exp(numeric_t(-x * x / 4));
  ;
}

/**
 * @brief polyval (Horner scheme)
 *
 * @param coeffs
 * @param x
 * @param N length of coeffs
 */
template <typename NUMERIC_T>
inline NUMERIC_T
polyval(NUMERIC_T* coeffs, NUMERIC_T x, int N)
{
  typedef NUMERIC_T numeric_t;
  numeric_t b = coeffs[N - 1];
  for (int i = 1; i < N - 1; ++i) {
    b = coeffs[N - 1 - i] + b * x;
  }
  return x * b + coeffs[0];
}

}  // end namespace detail

// ----------------------------------------------------------------------
/**
 * @brief Assemble shift matrix \f$ S^{\bar{x}}\f$.
 *
 * @tparam NUMERIC_T numeric type
 *
 */
template <typename NUMERIC_T>
class HShiftMatrix
{
 private:
  typedef NUMERIC_T numeric_t;
  typedef std::vector<numeric_t> vec_t;

 public:
  typedef Eigen::Matrix<numeric_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> matrix_t;

 public:
  /**
   *
   * @param N  max polynomial degree
   */
  HShiftMatrix(int N) { this->init(N); }

  HShiftMatrix() { /* empty */}

  /**
   * Compute polynomial coefficients of p(i,j;x)
   *
   */
  void init(int N);

  /**
   * Create linear Operator S^x by evaluating the polynomial
   *
   * @param x
   */
  void setx(numeric_t x);

  const matrix_t& get() const { return S_; }
  void dump(std::string fname) const;

 private:
  int size_;
  boost::multi_array<numeric_t, 3> coeffs_;
  /* compute coefficients of the S_-entry polynomial */
  detail::sentry<numeric_t> G_;
  matrix_t S_;
};

// ----------------------------------------------------------------------
template <typename NUMERIC_T>
void
HShiftMatrix<NUMERIC_T>::init(int N)
{
  size_ = N + 1;
  G_.init(N + 1);
  S_.resize(N + 1, N + 1);
  coeffs_.resize(boost::extents[size_][size_][2 * size_ - 1]);

  // std::fill(coeffs_.origin(), coeffs_.origin() + coeffs_.num_elements(), 0);
  for (int i = 0; i < size_; ++i) {
    for (int j = 0; j < size_; ++j) {
      for (int l = std::abs(i - j); l <= i + j; l += 2) {
        coeffs_[i][j][l] = G_.coeff(i, j, (i + j - l) / numeric_t(2));
      }
    }
  }
}

// ----------------------------------------------------------------------
template <typename NUMERIC_T>
void
HShiftMatrix<NUMERIC_T>::setx(numeric_t x)
{
  // Hint: polyval can be further optimized by using knowledge
  //       about which coefficients are zero.
  numeric_t f = ::math::exp(numeric_t(-x * x / 4));
  for (int i = 0; i < size_; ++i) {
    for (int j = 0; j < size_; ++j) {
      S_(i, j) = f * detail::polyval(coeffs_[i][j].origin(), x, coeffs_.shape()[2]);
    }
  }
}

// ----------------------------------------------------------------------
template <typename NUMERIC_T>
void
HShiftMatrix<NUMERIC_T>::dump(std::string fname) const
{
  std::ofstream fout(fname);
  fout << std::setprecision(10);
  fout << std::scientific;
  fout << S_;
  fout.close();
}

}  // end namespace boltzmann
