#pragma once

// system includes ---------------------------------------------------------
#include <boost/assert.hpp>
#include <boost/lexical_cast.hpp>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
// own includes ------------------------------------------------------------
#include "base/logger.hpp"
#include "ridgelet/ridgelet_cell_array.hpp"


class CG
{
 public:
  template <typename coeff_t, typename OPERATOR>
  bool solve(RidgeletCellArray<coeff_t> &x,
             const OPERATOR &A,
             const RidgeletCellArray<coeff_t> &b,
             const double reltol,
             const int maxit);

  template <typename DERIVED1, typename OPERATOR, typename DERIVED2>
  bool solve(Eigen::ArrayBase<DERIVED1> &x,
             const OPERATOR &A,
             const Eigen::ArrayBase<DERIVED2> &b,
             const double reltol,
             const int maxit);

  //@{
  /// status / log
  int iter() const { return iter_; }
  double relres() const { return relres_; }
  void set_log(bool log) { log_ = log; }
  void set_log_hist(bool t);
  //@}

 private:
  double relres_ = std::numeric_limits<double>::max();
  int iter_ = -1;
  bool log_ = false;
  bool log_hist_ = false;
};

template <typename coeff_t, typename OPERATOR>
bool
CG::solve(RidgeletCellArray<coeff_t> &x,
          const OPERATOR &A,
          const RidgeletCellArray<coeff_t> &b,
          const double reltol,
          const int maxit)
{
  typedef RidgeletCellArray<coeff_t> rca_t;
  typedef typename coeff_t::Scalar numeric_t;

  rca_t r(x.rf());
  r.resize(x);

  A.apply(r, x);
  r.sadd(-1, b, 1);

  double rho = r.norm();
  rho *= rho;
  if (rho < 1e-15) {
    relres_ = 0;
    iter_ = 0;
    if (log_)
      Logger::GetInstance() << (boost::lexical_cast<std::string>(iter_) + " " +
                                boost::lexical_cast<std::string>(relres_));
    return true;
  }

  rca_t p(x.rf());
  p.resize(x);

  rca_t Ap(x.rf());
  Ap.resize(x);

  p = r;
  for (iter_ = 1; iter_ <= maxit; ++iter_) {
    A.apply(Ap, p);
    // std::cout << "norm Ap: " << Ap.norm() << "\n";
    numeric_t alpha = rho / p.dot(Ap);
    x.sadd(1, p, alpha);
    r.sadd(1, Ap, -alpha);
    double old_rho = rho;
    double norm_r = r.norm();
    double norm_x = x.norm();
    //    std::cout << "norm_x: " << norm_x << "\n";
    // std::cout << "norm_r: " << norm_r << "\n";
    relres_ = norm_r / norm_x;
    if (log_ && log_hist_)
      Logger::GetInstance() << (boost::lexical_cast<std::string>(iter_) + " " +
                                boost::lexical_cast<std::string>(relres_));
    // std::cout << "CG::relres_ " << i << "\t" << std::setw(10) <<
    // std::setprecision(4) <<
    // std::scientific << relres_ << "\n";
    if (relres_ <= reltol) {
      // std::cout << "CG::Success after " << i << " iterations." << "\n";
      if (log_)
        Logger::GetInstance() << (boost::lexical_cast<std::string>(iter_) + " " +
                                  boost::lexical_cast<std::string>(relres_));
      return true;
    }

    rho = norm_r * norm_r;
    double beta = rho / old_rho;
    p.sadd(beta, r, 1);
  }
  return false;
}

template <typename DERIVED1, typename OPERATOR, typename DERIVED2>
bool
CG::solve(Eigen::ArrayBase<DERIVED1> &x,
          const OPERATOR &A,
          const Eigen::ArrayBase<DERIVED2> &b,
          const double reltol,
          const int maxit)
{
  BOOST_ASSERT_MSG(x.rows() == b.rows() && x.cols() == b.cols(), "dimension mismatch");

  static_assert(std::is_same<typename DERIVED1::Scalar, typename DERIVED2::Scalar>::value,
                "type mismatch");
  typedef typename DERIVED1::Scalar numeric_t;

  typedef Eigen::Array<numeric_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> array_t;

  array_t r(x.rows(), x.cols());

  A.apply(r, x.derived());
  r = b - r;
  double rho = std::sqrt(r.cwiseAbs2().sum());
  rho *= rho;
  if (rho < 1e-15) {
    relres_ = 0;
    iter_ = 0;
    if (log_)
      Logger::GetInstance() << (boost::lexical_cast<std::string>(iter_) + " " +
                                boost::lexical_cast<std::string>(relres_));
    return true;
  }

  array_t p(x.rows(), x.cols());
  array_t Ap(x.rows(), x.cols());

  p = r;
  for (iter_ = 1; iter_ <= maxit; ++iter_) {
    A.apply(Ap, p);
    // std::cout << "norm Ap: " << Ap.norm() << "\n";
    numeric_t alpha = rho / (p.conjugate() * Ap).sum();
    x += alpha * p;
    r -= alpha * Ap;
    double old_rho = rho;
    double norm_r = std::sqrt(r.cwiseAbs2().sum());
    double norm_x = std::sqrt(x.derived().cwiseAbs2().sum());
    //    std::cout << "norm_x: " << norm_x << "\n";
    // std::cout << "norm_r: " << norm_r << "\n";
    relres_ = norm_r / norm_x;
    if (log_ && log_hist_)
      Logger::GetInstance() << (boost::lexical_cast<std::string>(iter_) + " " +
                                boost::lexical_cast<std::string>(relres_));
    if (relres_ <= reltol) {
      if (log_)
        Logger::GetInstance() << (boost::lexical_cast<std::string>(iter_) + " " +
                                  boost::lexical_cast<std::string>(relres_));
      return true;
    }

    rho = norm_r * norm_r;
    double beta = rho / old_rho;
    p = beta * p + r;
  }
  return false;
}

void
CG::set_log_hist(bool t)
{
  if (t == true) log_ = true;
  log_hist_ = t;
}
