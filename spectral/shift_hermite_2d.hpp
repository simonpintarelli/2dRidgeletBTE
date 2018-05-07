#pragma once

// system includes --------------------------------------------------
#include <Eigen/Core>
#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>
#include <vector>
// own includes -----------------------------------------------------
#include "aux/message.hpp"
#include "aux/timer.hpp"
#include "shift_hermite.hpp"

namespace boltzmann {

template <typename BASIS, typename NUMERIC = double>
class ShiftHermite2D
{
 public:
  typedef NUMERIC numeric_t;
  typedef BASIS basis_t;
  typedef std::vector<numeric_t> std_vec_t;

 public:
  /**
   *
   *
   * @param basis
   * @param N      max. polynomial degree+1 in one variable
   */
  ShiftHermite2D(const basis_t& basis);

  void init();

  void shift(numeric_t* c, numeric_t x, numeric_t y);

  // debug
  const HShiftMatrix<numeric_t>& get_sx() { return sx_op_; }
  const HShiftMatrix<numeric_t>& get_sy() { return sy_op_; }

 private:
  typedef typename BASIS::elem_t elem_t;
  typedef typename boost::mpl::at_c<typename elem_t::types_t, 0>::type hx_t;
  typedef typename boost::mpl::at_c<typename elem_t::types_t, 1>::type hy_t;
  typedef Eigen::Matrix<numeric_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> matrix_t;

 private:
  const basis_t& basis_;
  /// buffers
  matrix_t buf_in;
  matrix_t buf_out;
  /// permutation vector
  std::vector<unsigned int> perm_;
  bool is_initialized_;

  typename elem_t::Acc::template get<hx_t> get_hx;
  typename elem_t::Acc::template get<hy_t> get_hy;

  HShiftMatrix<numeric_t> sx_op_;
  HShiftMatrix<numeric_t> sy_op_;
};

// ----------------------------------------------------------------------
template <typename BASIS, typename NUMERIC>
ShiftHermite2D<BASIS, NUMERIC>::ShiftHermite2D(const basis_t& basis)
    : basis_(basis)
    ,

    perm_(basis.n_dofs())
    , is_initialized_(false)
{ /* empty */
}

// ----------------------------------------------------------------------
template <typename BASIS, typename NUMERIC>
void
ShiftHermite2D<BASIS, NUMERIC>::init()
{
  // max degree
  unsigned int max_kx =
      get_hx(*std::max_element(basis_.begin(),
                               basis_.end(),
                               [&](const elem_t& e1, const elem_t& e2) {
                                 return get_hx(e1).get_id().k < get_hx(e2).get_id().k;
                               }))
          .get_id()
          .k;
  unsigned int max_ky =
      get_hy(*std::max_element(basis_.begin(),
                               basis_.end(),
                               [&](const elem_t& e1, const elem_t& e2) {
                                 return get_hy(e1).get_id().k < get_hy(e2).get_id().k;
                               }))
          .get_id()
          .k;
  buf_in.resize(max_kx + 1, max_ky + 1);
  buf_out.resize(max_kx + 1, max_ky + 1);

  buf_in.fill(0);
  buf_out.fill(0);

  unsigned int stride = buf_in.cols();

  // build permutation vector
  unsigned int i = 0;
  for (auto elem = basis_.begin(); elem < basis_.end(); ++elem, ++i) {
    unsigned int kx = get_hx(*elem).get_id().k;
    unsigned int ky = get_hy(*elem).get_id().k;
    perm_[i] = kx * stride + ky;
  }

  Timer timer;
  // initialize shift matrices
  timer.restart();
  sx_op_.init(max_kx);
  sy_op_.init(max_ky);
  print_timer(timer.stop(), "initialize shift matrices");
  is_initialized_ = true;
}

// ----------------------------------------------------------------------
template <typename BASIS, typename NUMERIC>
void
ShiftHermite2D<BASIS, NUMERIC>::shift(numeric_t* c, numeric_t x, numeric_t y)
{
  assert(is_initialized_);

  // permute
  numeric_t* in = buf_in.data();

  for (unsigned int i = 0; i < basis_.n_dofs(); ++i) {
    in[perm_[i]] = c[i];
  }

  // apply shift matrices
  sx_op_.setx(x);
  sy_op_.setx(y);

  auto& Sx = sx_op_.get();
  auto& Sy = sy_op_.get();

  buf_out = Sx * buf_in * Sy.transpose();

  numeric_t* out = buf_out.data();

  // invert permutation and overwrite c
  for (unsigned int i = 0; i < basis_.n_dofs(); ++i) {
    c[i] = out[perm_[i]];
  }
}

}  // end namespace boltzmann
