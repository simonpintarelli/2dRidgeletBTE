#pragma once

#include "cg.hpp"
#include "operators/operators.hpp"
#include "ridgelet/ridgelet_cell_array.hpp"


template <typename STORAGE = Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
class RidgeletSolver
{
 public:
  typedef STORAGE coeff_t;
  typedef RidgeletCellArray<coeff_t> rca_t;

 public:
  template <typename FRAME>
  RidgeletSolver(const FRAME &frame, double vx, double vy, double dt = 1)
      : bp_(frame)
      , Dinv_(make_inv_diagonal_preconditioner(frame, dt * vx, dt * vy))
  {
    D_ = Dinv_;
    D_.invert();
    /* empty  */
  }

  RidgeletSolver() {}

  /**
   *
   * @param x     dst / starting vector
   * @param B     Matrix (operator)
   * @param b     right hand side
   * @param tol   cg tol
   * @param maxit cg maxit
   * @param log   store relres?
   *
   */
  template <typename OPERATOR>
  void solve(rca_t &x, OPERATOR &B, const rca_t &b, double tol, int maxit);

  template <typename OPERATOR>
  void solve(rca_t &x, OPERATOR &B, const rca_t &b, double tol, int maxit, bool log);

  void set_log(bool f) { cg_.set_log(f); }
  double relres() const { return cg_.relres(); }
  int iter() const { return cg_.iter(); }

 private:
  // preconditioned b
  rca_t bp_;
  CG cg_;
  DiagonalOperator<coeff_t> D_;
  DiagonalOperator<coeff_t> Dinv_;
};

template <typename STORAGE>
template <typename OPERATOR>
void
RidgeletSolver<STORAGE>::solve(
    rca_t &x, OPERATOR &B, const rca_t &b, double tol, int maxit, bool log)
{
  D_.apply(bp_, b);
  Dinv_.apply(x);
  cg_.set_log(log);
  cg_.solve(x, B, bp_, tol, maxit);

  D_.apply(x);
}

template <typename STORAGE>
template <typename OPERATOR>
void
RidgeletSolver<STORAGE>::solve(rca_t &x, OPERATOR &B, const rca_t &b, double tol, int maxit)
{
  D_.apply(bp_, b);
  Dinv_.apply(x);
  cg_.solve(x, B, bp_, tol, maxit);

  D_.apply(x);
}

// --------------------------------------------------------------------------------
// --------------------------------------------------------------------------------
// --------------------------------------------------------------------------------

template <typename STORAGE = Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
class RidgeletSolverNOP
{
 public:
  typedef STORAGE coeff_t;
  typedef RidgeletCellArray<coeff_t> rca_t;

 public:
  template <typename FRAME>
  RidgeletSolverNOP(const FRAME &frame, double vx, double vy, double dt)
      : bp_(frame)
  {
  }

  RidgeletSolverNOP() {}

  /**
   *
   * @param x     dst / starting vector
   * @param B     Matrix (operator)
   * @param b     right hand side
   * @param tol   cg tol
   * @param maxit cg maxit
   * @param log   store relres?
   *
   */
  template <typename OPERATOR>
  void solve(rca_t &x, OPERATOR &B, const rca_t &b, double tol, int maxit);

  template <typename OPERATOR>
  void solve(rca_t &x, OPERATOR &B, const rca_t &b, double tol, int maxit, bool log);

  void set_log(bool f) { cg_.set_log(f); }
  double relres() const { return cg_.relres(); }
  int iter() const { return cg_.iter(); }

 private:
  // preconditioned b
  rca_t bp_;
  CG cg_;
};

template <typename STORAGE>
template <typename OPERATOR>
void
RidgeletSolverNOP<STORAGE>::solve(
    rca_t &x, OPERATOR &B, const rca_t &b, double tol, int maxit, bool log)
{
  cg_.set_log(log);
  cg_.solve(x, B, b, tol, maxit);
}

template <typename STORAGE>
template <typename OPERATOR>
void
RidgeletSolverNOP<STORAGE>::solve(rca_t &x, OPERATOR &B, const rca_t &b, double tol, int maxit)
{
  cg_.solve(x, B, b, tol, maxit);
}
