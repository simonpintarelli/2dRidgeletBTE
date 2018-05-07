#pragma once

#include <boost/math/constants/constants.hpp>

#include "fft/ft_grid_helpers.hpp"


enum class derivative_t
{
  dX,
  dY
};

template <typename DERIVED1, typename DERIVED2>
double
compute_mass_entry(const Eigen::SparseMatrixBase<DERIVED1> &m1,
                   const Eigen::SparseMatrixBase<DERIVED2> &m2)
{
  int n = std::max(m1.rows(), m2.rows());
  int m = std::max(m1.cols(), m2.cols());

  typedef Eigen::SparseMatrix<double, Eigen::RowMajor> sp_mat_t;
  sp_mat_t b1, b2;
  ftpad(b1, m1, n, m);
  ftpad(b2, m2, n, m);

  double sum = 0;

  for (int ii = 0; ii < b1.outerSize(); ++ii) {
    sp_mat_t::InnerIterator it2(b2, ii);
    for (sp_mat_t::InnerIterator it1(b1, ii); it1; ++it1) {
      while (it2.col() < it1.col()) ++it2;
      if (!bool(it2))
        break;
      else if (it2.col() == it1.col())
        sum += it2.value() * it1.value();
      else
        continue;
    }
  }
  return sum;
}

/**
 * \f$ computes ... \f$
 *
 * @param m1
 * @param m2
 * @param tdx
 * @param tdy
 *
 * @return
 */
template <typename DERIVED1, typename DERIVED2>
double
compute_mass_entry(const Eigen::SparseMatrixBase<DERIVED1> &m1,
                   const Eigen::SparseMatrixBase<DERIVED2> &m2,
                   double tdx,
                   double tdy)
{
  const double PI = boost::math::constants::pi<double>();

  int n = std::max(m1.rows(), m2.rows());
  int m = std::max(m1.cols(), m2.cols());

  typedef Eigen::SparseMatrix<double, Eigen::RowMajor> sp_mat_t;
  sp_mat_t b1, b2;
  ftpad(b1, m1, n, m);
  ftpad(b2, m2, n, m);

  double sum = 0;

  for (int ii = 0; ii < b1.outerSize(); ++ii) {
    int yhat = to_freq(ii, n);
    sp_mat_t::InnerIterator it2(b2, ii);
    for (sp_mat_t::InnerIterator it1(b1, ii); it1; ++it1) {
      while (bool(it2) && it2.col() < it1.col()) ++it2;
      if (!bool(it2))
        break;
      else if (it2.col() == it1.col()) {
        int xhat = to_freq(it2.col(), m);
        const double f = std::cos(2 * PI * (tdy * yhat + tdx * xhat));
        sum += it2.value() * it1.value() * f;
      }

      else
        continue;
    }
  }
  return sum;
}

/**
 * \f$ \int_{L_x, L_y} \partial_k r_{i1}(x) \partial_l r_{i2}(x) \mathrm{d}x \f$
 *
 * @param m1 Fourier coefficients corresponding to r_{i1}
 * @param m2 Fourier coefficients corresponding to r_{i2}
 * @param tdx translation grid difference vector in x
 * @param tdy translation grid difference vector in y
 * @param dm1 derivative applied to \f$r_{i1}\f$
 * @param dm2 derivative applied to \f$r_{i2}\f$
 *
 * @return
 */
template <typename DERIVED1, typename DERIVED2>
double
compute_tentry(const Eigen::SparseMatrixBase<DERIVED1> &m1,
               const Eigen::SparseMatrixBase<DERIVED2> &m2,
               double tdx,
               double tdy,
               derivative_t dm1,
               derivative_t dm2,
               double Lx = 1.0,
               double Ly = 1.0)
{
  const double PI = boost::math::constants::pi<double>();
  const double PI2 = PI * PI;

  int n = std::max(m1.rows(), m2.rows());
  int m = std::max(m1.cols(), m2.cols());

  typedef Eigen::SparseMatrix<double, Eigen::RowMajor> sp_mat_t;
  sp_mat_t b1, b2;
  ftpad(b1, m1, n, m);
  ftpad(b2, m2, n, m);

  double sum = 0;
  if (dm1 == derivative_t::dX && dm2 == derivative_t::dX) {
    for (int ii = 0; ii < b1.outerSize(); ++ii) {
      int yhat = to_freq(ii, n);
      sp_mat_t::InnerIterator it2(b2, ii);
      for (sp_mat_t::InnerIterator it1(b1, ii); it1; ++it1) {
        while (bool(it2) && it2.col() < it1.col()) ++it2;
        if (!bool(it2))
          break;
        else if (it2.col() == it1.col()) {
          int xhat = to_freq(it2.col(), m);
          const double f = std::cos(2 * PI * (tdy * yhat + tdx * xhat));
          sum += xhat * xhat / (Lx * Lx) * it2.value() * it1.value() * f;
        } else
          continue;
      }
    }
  } else if (dm1 == derivative_t::dY && dm2 == derivative_t::dY) {
    for (int ii = 0; ii < b1.outerSize(); ++ii) {
      int yhat = to_freq(ii, n);
      sp_mat_t::InnerIterator it2(b2, ii);
      for (sp_mat_t::InnerIterator it1(b1, ii); it1; ++it1) {
        while (bool(it2) && it2.col() < it1.col()) ++it2;
        if (!bool(it2))
          break;
        else if (it2.col() == it1.col()) {
          int xhat = to_freq(it2.col(), m);
          const double f = std::cos(2 * PI * (tdy * yhat + tdx * xhat));
          sum += yhat * yhat / (Ly * Ly) * it2.value() * it1.value() * f;
        } else
          continue;
      }
    }
  } else {
    // dm1 == dY and dm2 == dX or vice versa
    for (int ii = 0; ii < b1.outerSize(); ++ii) {
      int yhat = to_freq(ii, n);
      sp_mat_t::InnerIterator it2(b2, ii);
      for (sp_mat_t::InnerIterator it1(b1, ii); it1; ++it1) {
        while (bool(it2) && it2.col() < it1.col()) ++it2;
        if (!bool(it2))
          break;
        else if (it2.col() == it1.col()) {
          int xhat = to_freq(it2.col(), m);
          const double f = std::cos(2 * PI * (tdy * yhat + tdx * xhat));
          sum += xhat * yhat / (Lx * Ly) * it2.value() * it1.value() * f;
        } else
          continue;
      }
    }
  }

  return 4 * PI2 * sum;
}
