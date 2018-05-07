#pragma once

#include <Eigen/Dense>
#include <algorithm>

// own includes -------------------------------------------------
#include "base/array_buffer.hpp"
#include "h2n_1d.hpp"

namespace boltzmann {
template <typename BASIS, typename NUMERIC_T = double>
class Hermite2Nodal
{
 public:
  typedef NUMERIC_T numeric_t;
  typedef Eigen::Matrix<numeric_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> matrix_t;

 public:
  /**
   *
   *
   * @param basis  Hermite basis
   * @param K      max poly. degree
   * @param init   functor to initialize the 1d trafo matrix (initializers can
   * be found in
   * h2n_1d.hpp)
   */
  template <typename INITIALIZER>
  Hermite2Nodal(const BASIS &basis, const int K, const INITIALIZER &init);

  /**
   * @brief Transform from Nodal to Hermite basis.
   *
   * Also see @ref to_nodal ,
   *
   * @param dst Hermite coefficients ordered according to @ref basis.
   * @param src Nodal coefficients, \f$ src(i,j) = f(x_j, y_i) \f$.
   * @param transpose defaults to false, true is ``experimental''
   *
   *
   */
  template <typename MATRIX>
  void to_hermite(numeric_t *dst, const MATRIX &src, bool transpose = false) const;

  template <typename DERIVED, typename DERIVED2>
  void to_hermite(Eigen::DenseBase<DERIVED> &dst,
                  const Eigen::DenseBase<DERIVED2> &src,
                  bool transpose = false) const;

  /**
   * @brief Transform from Hermite to Nodal basis.
   *
   * @param dst Output Nodal coefficients dst(i,j) correpsonds to nodes \f$ x_i,
   * y_j\f$.
   * @param c   Hermite coefficients in @ref basis
   * @param tranpose: defaults to false, true is ``experimental''
   *
   *
   * Internally the Hermite coefficients are arranged like so:
   * @remark{
   *
   *     H[deg(y), deg(x)] = c_(iy, ix)
   * }
   *
   */
  template <typename MATRIX>
  void to_nodal(MATRIX &dst, const numeric_t *c, bool transpose = false) const;

  template <typename DERIVED, typename DERIVED2>
  void to_nodal(Eigen::DenseBase<DERIVED> &dst,
                const Eigen::DenseBase<DERIVED2> &src,
                bool transpose = false) const;

  const matrix_t &get_n2h() const { return N2H_; }
  const matrix_t &get_h2n() const { return H2N_; }

 private:
  std::vector<unsigned int> perm_;
  int K_;

  matrix_t N2H_;
  matrix_t H2N_;
  thread_local static ::ArrayBuffer<> buf_;

  BASIS hbasis_;
};

template <typename BASIS, typename NUMERIC_T>
thread_local ::ArrayBuffer<> Hermite2Nodal<BASIS, NUMERIC_T>::buf_;

// ----------------------------------------------------------------------
template <typename BASIS, typename NUMERIC_T>
template <typename INITIALIZER>
Hermite2Nodal<BASIS, NUMERIC_T>::Hermite2Nodal(const BASIS &basis,
                                               const int K,
                                               const INITIALIZER &init)
    : K_(K)
    , hbasis_(basis)
{
  buf_.reserve(K * K);
  // initialize the 1d transformation matrix
  init(H2N_, N2H_);

  typedef typename BASIS::elem_t elem_t;
  typedef typename boost::mpl::at_c<typename elem_t::types_t, 0>::type hx_t;
  typedef typename boost::mpl::at_c<typename elem_t::types_t, 1>::type hy_t;

  typename elem_t::Acc::template get<hx_t> get_hx;
  typename elem_t::Acc::template get<hy_t> get_hy;

  // max degree
  unsigned int max_kx =
      get_hx(*std::max_element(basis.begin(),
                               basis.end(),
                               [&](const elem_t &e1, const elem_t &e2) {
                                 return get_hx(e1).get_id().k < get_hx(e2).get_id().k;
                               }))
          .get_id()
          .k;
  unsigned int max_ky =
      get_hy(*std::max_element(basis.begin(),
                               basis.end(),
                               [&](const elem_t &e1, const elem_t &e2) {
                                 return get_hy(e1).get_id().k < get_hy(e2).get_id().k;
                               }))
          .get_id()
          .k;

  assert(max_kx == max_ky);

  unsigned int stride = K;
  perm_.resize(hbasis_.n_dofs());
  // build permutation vector
  unsigned int i = 0;
  for (auto elem = basis.begin(); elem < basis.end(); ++elem, ++i) {
    unsigned int kx = get_hx(*elem).get_id().k;
    unsigned int ky = get_hy(*elem).get_id().k;
    perm_[i] = kx * stride + ky;
  }
}

// ----------------------------------------------------------------------
template <typename BASIS, typename NUMERIC_T>
template <typename MATRIX>
void
Hermite2Nodal<BASIS, NUMERIC_T>::to_nodal(MATRIX &dst, const numeric_t *c, bool transpose) const
{
  auto TMP = buf_.get<matrix_t>(K_, K_);

  if (!transpose) {
    TMP.fill(0);
    numeric_t *data = TMP.data();
    for (unsigned int i = 0; i < hbasis_.n_dofs(); ++i) {
      data[perm_[i]] = c[i];
    }
    //  const auto& T = h2n_.get_matrix();
    const auto &T = H2N_;
    dst = T * TMP * T.transpose();
  } else {
    // ATTENTION: untested!
    TMP.fill(0);
    numeric_t *data = TMP.data();
    for (unsigned int i = 0; i < hbasis_.n_dofs(); ++i) {
      data[perm_[i]] = c[i];
    }
    const auto &T = H2N_;
    dst = T.transpose() * TMP * T;
  }
}

// -----------------------------------------------------------------------
template <typename BASIS, typename NUMERIC_T>
template <typename DERIVED, typename DERIVED2>
void
Hermite2Nodal<BASIS, NUMERIC_T>::to_nodal(Eigen::DenseBase<DERIVED> &dst,
                                          const Eigen::DenseBase<DERIVED2> &c,
                                          bool transpose) const
{
  auto TMP = buf_.get<matrix_t>(K_, K_);

  if (!transpose) {
    TMP.fill(0);
    numeric_t *data = TMP.data();
    for (unsigned int i = 0; i < hbasis_.n_dofs(); ++i) {
      data[perm_[i]] = c[i];
    }
    //  const auto& T = h2n_.get_matrix();
    const auto &T = H2N_;
    dst = T * TMP * T.transpose();
  } else {
    // ATTENTION: untested!
    TMP.fill(0);
    numeric_t *data = TMP.data();
    for (unsigned int i = 0; i < hbasis_.n_dofs(); ++i) {
      data[perm_[i]] = c[i];
    }
    const auto &T = H2N_;
    dst = T.transpose() * TMP * T;
  }
}

// ----------------------------------------------------------------------
template <typename BASIS, typename NUMERIC_T>
template <typename MATRIX>
void
Hermite2Nodal<BASIS, NUMERIC_T>::to_hermite(numeric_t *dst, const MATRIX &src, bool transpose) const
{
  auto TMP = buf_.get<matrix_t>(K_, K_);
  if (!transpose) {
    const auto &T = N2H_;
    TMP = T * src * T.transpose();
    // undo permuation
    for (unsigned int i = 0; i < hbasis_.n_dofs(); ++i) {
      dst[i] = TMP.data()[perm_[i]];
    }
  } else {
    // ATTENTION: untested!
    const auto &T = N2H_;
    TMP = T.transpose() * src * T;

    numeric_t *data = TMP.data();
    for (unsigned int i = 0; i < hbasis_.n_dofs(); ++i) {
      dst[i] = data[perm_[i]];
    }
  }
}

// ----------------------------------------------------------------------
template <typename BASIS, typename NUMERIC_T>
template <typename DERIVED, typename DERIVED2>
void
Hermite2Nodal<BASIS, NUMERIC_T>::to_hermite(Eigen::DenseBase<DERIVED> &dst,
                                            const Eigen::DenseBase<DERIVED2> &src,
                                            bool transpose) const
{
  auto TMP = buf_.get<matrix_t>(K_, K_);
  if (!transpose) {
    const auto &T = N2H_;
    TMP = T * src.derived() * T.transpose();
    // undo permuation
    for (unsigned int i = 0; i < hbasis_.n_dofs(); ++i) {
      dst[i] = TMP.data()[perm_[i]];
    }
  } else {
    // ATTENTION: untested!
    const auto &T = N2H_;
    TMP = T.transpose() * src.derived() * T;

    numeric_t *data = TMP.data();
    for (unsigned int i = 0; i < hbasis_.n_dofs(); ++i) {
      dst[i] = data[perm_[i]];
    }
  }
}

}  // end namespace boltzmann
