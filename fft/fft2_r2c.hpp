#pragma once

#include <fftw3.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <boost/assert.hpp>
#include <complex>
#include <type_traits>

#include "base/array_buffer.hpp"
#include "planner.hpp"
#include "shift.hpp"


namespace local_ {
template <typename D1, typename D2>
struct enable_if_cc
    : public std::enable_if<std::is_same<typename D1::Scalar, std::complex<double>>::value &&
                            std::is_same<typename D2::Scalar, std::complex<double>>::value>
{
};

template <typename D1, typename D2>
struct enable_if_cr
    : public std::enable_if<std::is_same<typename D1::Scalar, std::complex<double>>::value &&
                            std::is_same<typename D2::Scalar, double>::value>
{
};

template <typename D1, typename D2>
struct enable_if_rc
    : public std::enable_if<std::is_same<typename D1::Scalar, double>::value &&
                            std::is_same<typename D2::Scalar, std::complex<double>>::value>
{
};
}  // local_

template <typename PLAN_HANDLER>
class FFTr2c
{
 public:
  typedef Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> array_t;
  typedef Eigen::Array<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      complex_array_t;

  typedef std::complex<double> cdouble;

 public:
  //@{
  /// real valued transforms

  /**
   * @brief forward transform (real-valued)
   *
   * @param[out] dst
   * @param[in]  src
   * @param[in]  scale if true: scales output by 1/numel(src)
   *
   */
  template <typename DERIVED1, typename DERIVED2>
  typename local_::enable_if_cr<DERIVED1, DERIVED2>::type fft2(
      Eigen::DenseBase<DERIVED1> &dst,
      const Eigen::DenseBase<DERIVED2> &src,
      bool scale = true) const;

  /**
   * @brief inverse transform (does not preserve input, real-valued)
   *
   * @param[out] dst
   * @param[in]  src   full spectrum
   */
  template <typename DERIVED1, typename DERIVED2>
  typename local_::enable_if_rc<DERIVED1, DERIVED2>::type ifft2(
      Eigen::DenseBase<DERIVED1> &dst, Eigen::DenseBase<DERIVED2> &src) const;

  /**
   * @brief 2-dim fft (including fftshift, real-valued)
   *
   * @param[out] dst  complex array (centered zero-frequency convention)
   * @param[in] src   real array
   * @param[in] scale if true: scales output by 1/numel(src)
   *
   * This is the inverse of ift for \var scale set to false!
   *
   */
  template <typename DERIVED1, typename DERIVED2>
  typename local_::enable_if_cr<DERIVED1, DERIVED2>::type ft(Eigen::DenseBase<DERIVED1> &dst,
                                                             const Eigen::DenseBase<DERIVED2> &src,
                                                             bool scale = true) const;

  /**
   * @brief ifft2 (including ifftshift, real-valued)
   *
   * @param[out] dst   real array
   * @param[in]  src   complex array
   */
  template <typename DERIVED1, typename DERIVED2>
  typename local_::enable_if_rc<DERIVED1, DERIVED2>::type ift(
      Eigen::DenseBase<DERIVED1> &dst, const Eigen::DenseBase<DERIVED2> &src) const;
  //@}

  //@{
  /// complex transforms
  /**
   * complex, complex
   * @brief fft2 (including fftshift)
   *
   * @param[out] dst (in centered zero-frequency convention)
   * @param[int] src
   * @param bool  if true: scales output by 1/numel(src)
   *
   * This is the inverse of ift for \var scale set to false.
   *
   */
  template <typename DERIVED1, typename DERIVED2>
  typename local_::enable_if_cc<DERIVED1, DERIVED2>::type ft(Eigen::DenseBase<DERIVED1> &dst,
                                                             const Eigen::DenseBase<DERIVED2> &src,
                                                             bool scale = true) const;

  /**
   * complex, complex
   * @brief complex ifft2 (including ifftshift)
   *
   * @param[out] dst
   * @param[in] src
   */
  template <typename DERIVED1, typename DERIVED2>
  typename local_::enable_if_cc<DERIVED1, DERIVED2>::type ift(
      Eigen::DenseBase<DERIVED1> &dst, const Eigen::DenseBase<DERIVED2> &src) const;

  /**
   *
   */
  template <typename DERIVED1, typename DERIVED2>
  typename local_::enable_if_cc<DERIVED1, DERIVED2>::type fft2(
      Eigen::DenseBase<DERIVED1> &dst,
      const Eigen::DenseBase<DERIVED2> &src,
      bool scale = true) const;

  /**
   * @brief ifft2 (c -> c)
   *
   * @param[out] dst  destination
   * @param[in] src  will be overwritten by FFTW!
   */
  template <typename DERIVED1, typename DERIVED2>
  typename local_::enable_if_cc<DERIVED1, DERIVED2>::type ifft2(
      Eigen::DenseBase<DERIVED1> &dst, Eigen::DenseBase<DERIVED2> &src) const;
  //@}

  PLAN_HANDLER &get_plan() { return plan_h_; }

 private:
  static PLAN_HANDLER plan_h_;
  thread_local static ArrayBuffer<> buf_;
};

template <typename PLAN_HANDLER>
thread_local ArrayBuffer<> FFTr2c<PLAN_HANDLER>::buf_;

template <typename PLAN_HANDLER>
PLAN_HANDLER FFTr2c<PLAN_HANDLER>::plan_h_;

// --------------------------------------------------------------------------------
// --------------------------------------------------------------------------------
// -------------------------- REAL-COMPLEX TRANSFORMS
// -----------------------------
// --------------------------------------------------------------------------------
// --------------------------------------------------------------------------------
template <typename PLAN_HANDLER>
template <typename DERIVED1, typename DERIVED2>
typename local_::enable_if_cr<DERIVED1, DERIVED2>::type
FFTr2c<PLAN_HANDLER>::fft2(Eigen::DenseBase<DERIVED1> &dst,
                           const Eigen::DenseBase<DERIVED2> &src,
                           bool scale) const
{
  static_assert(DERIVED1::IsRowMajor, "requires row-major storage");
  static_assert(DERIVED2::IsRowMajor, "requires row-major storage");
  static_assert(sizeof(fftw_complex) == sizeof(cdouble), "type mismatch");
  // assert(dst.rows() == src.rows());
  // assert(dst.cols() == src.cols());

  dst.derived().resize(src.rows(), src.cols());

  // typedef double fftw_cdouble[2];
  typedef fftw_complex fftw_cdouble;
  const int n0 = src.rows();
  const int n1 = src.cols();
  int n[2] = {n0, n1};

  fftw_cdouble *out = reinterpret_cast<fftw_cdouble *>(dst.derived().data());
  double *in = const_cast<double *>(src.derived().data());

  fftw_plan fwd_plan = plan_h_.get_plan(n, PLAN_HANDLER::FWD, ft_type::R2C);
  BOOST_ASSERT_MSG(fwd_plan != NULL, "fftw plan not found!");
  fftw_execute_dft_r2c(fwd_plan, in, out);

  // mirror coefficients
  for (int i = 0; i < n0; ++i) {
    int idest = (n0 - i) % n0;
    for (int j = 1; j < n1 / 2 + n1 % 2; ++j) {
      dst(idest, n1 - j) = std::conj(dst(i, j));
    }
  }
  if (scale) {
    double f = 1. / (n0 * n1);
    dst *= f;
  }
}

// --------------------------------------------------------------------------------
template <typename PLAN_HANDLER>
template <typename DERIVED1, typename DERIVED2>
typename local_::enable_if_rc<DERIVED1, DERIVED2>::type
FFTr2c<PLAN_HANDLER>::ifft2(Eigen::DenseBase<DERIVED1> &dst, Eigen::DenseBase<DERIVED2> &src) const
{
  static_assert(std::is_same<typename DERIVED2::Scalar, std::complex<double>>::value,
                "type mismatch");
  static_assert(std::is_same<typename DERIVED1::Scalar, double>::value, "type mismatch");
  static_assert(sizeof(fftw_complex) == sizeof(cdouble), "type mismatch");
  static_assert(DERIVED1::IsRowMajor, "requires row-major storage");
  static_assert(DERIVED2::IsRowMajor, "requires row-major storage");

  dst.derived().resize(src.rows(), src.cols());
  // typedef double fftw_cdouble[2];
  typedef fftw_complex fftw_cdouble;
  const int n0 = src.rows();
  const int n1 = src.cols();
  int n[2] = {n0, n1};

  fftw_cdouble *in = reinterpret_cast<fftw_cdouble *>(src.derived().data());
  double *out = dst.derived().data();

  fftw_plan inv_plan = plan_h_.get_plan(n, PLAN_HANDLER::INV, ft_type::R2C);
  BOOST_ASSERT_MSG(inv_plan != NULL, "fftw plan not found!");
  // fftw_execute(inv_plan);
  fftw_execute_dft_c2r(inv_plan, in, out);

  double f = 1. / (n0 * n1);
  dst *= f;
}

// --------------------------------------------------------------------------------
template <typename PLAN_HANDLER>
template <typename DERIVED1, typename DERIVED2>
typename local_::enable_if_cr<DERIVED1, DERIVED2>::type
FFTr2c<PLAN_HANDLER>::ft(Eigen::DenseBase<DERIVED1> &dst,
                         const Eigen::DenseBase<DERIVED2> &src,
                         bool scale) const
{
  static_assert(std::is_same<typename DERIVED1::Scalar, std::complex<double>>::value,
                "type mismatch");
  static_assert(std::is_same<typename DERIVED2::Scalar, double>::value, "type mismatch");

  auto tmp = buf_.get<complex_array_t>(src.rows(), src.cols());
  this->fft2(tmp, src, scale);
  dst.resize(tmp.rows(), tmp.cols());
  fftshift(dst, tmp);
}

// --------------------------------------------------------------------------------
template <typename PLAN_HANDLER>
template <typename DERIVED1, typename DERIVED2>
typename local_::enable_if_rc<DERIVED1, DERIVED2>::type
FFTr2c<PLAN_HANDLER>::ift(Eigen::DenseBase<DERIVED1> &dst,
                          const Eigen::DenseBase<DERIVED2> &src) const
{
  static_assert(std::is_same<typename DERIVED1::Scalar, double>::value, "type mismatch");
  static_assert(std::is_same<typename DERIVED2::Scalar, std::complex<double>>::value,
                "type mismatch");

  auto tmp = buf_.get<complex_array_t>(src.rows(), src.cols());
  ifftshift(tmp, src);
  this->ifft2(dst, tmp);
}

// --------------------------------------------------------------------------------
// --------------------------------------------------------------------------------
// ------------------------ COMPLEX-COMPLEX TRANSFORMS ----------------------------
// --------------------------------------------------------------------------------
// --------------------------------------------------------------------------------
template <typename PLAN_HANDLER>
template <typename DERIVED1, typename DERIVED2>
typename local_::enable_if_cc<DERIVED1, DERIVED2>::type
FFTr2c<PLAN_HANDLER>::fft2(Eigen::DenseBase<DERIVED1> &dst,
                           const Eigen::DenseBase<DERIVED2> &src,
                           bool scale) const
{
  static_assert(DERIVED1::IsRowMajor, "requires row-major storage");
  static_assert(DERIVED2::IsRowMajor, "requires row-major storage");
  static_assert(sizeof(fftw_complex) == sizeof(cdouble), "type mismatch");

  dst.derived().resize(src.rows(), src.cols());

  typedef fftw_complex fftw_cdouble;
  const int n0 = src.rows();
  const int n1 = src.cols();
  int n[2] = {n0, n1};

  fftw_cdouble *in =
      const_cast<fftw_cdouble *>(reinterpret_cast<const fftw_cdouble *>(src.derived().data()));
  fftw_cdouble *out = reinterpret_cast<fftw_cdouble *>(dst.derived().data());

  fftw_plan fwd_plan = plan_h_.get_plan(n, PLAN_HANDLER::FWD, ft_type::C2C);
  BOOST_ASSERT_MSG(fwd_plan != NULL, "fftw plan not found!");
  fftw_execute_dft(fwd_plan, in, out);

  if (scale) {
    double f = 1. / (n0 * n1);
    dst *= f;
  }
}

// --------------------------------------------------------------------------------
template <typename PLAN_HANDLER>
template <typename DERIVED1, typename DERIVED2>
typename local_::enable_if_cc<DERIVED1, DERIVED2>::type
FFTr2c<PLAN_HANDLER>::ifft2(Eigen::DenseBase<DERIVED1> &dst, Eigen::DenseBase<DERIVED2> &src) const
{
  static_assert(sizeof(fftw_complex) == sizeof(cdouble), "type mismatch");
  static_assert(DERIVED1::IsRowMajor, "requires row-major storage");
  static_assert(DERIVED2::IsRowMajor, "requires row-major storage");

  dst.derived().resize(src.rows(), src.cols());
  // typedef double fftw_cdouble[2];
  typedef fftw_complex fftw_cdouble;
  const int n0 = src.rows();
  const int n1 = src.cols();
  int n[2] = {n0, n1};

  fftw_cdouble *in =
      const_cast<fftw_cdouble *>(reinterpret_cast<const fftw_cdouble *>(src.derived().data()));
  fftw_cdouble *out = reinterpret_cast<fftw_cdouble *>(dst.derived().data());

  fftw_plan inv_plan = plan_h_.get_plan(n, PLAN_HANDLER::INV, ft_type::C2C);
  BOOST_ASSERT_MSG(inv_plan != NULL, "fftw plan not found!");
  // fftw_execute(inv_plan);
  fftw_execute_dft(inv_plan, in, out);

  double f = 1. / (n0 * n1);
  dst *= f;
}

// --------------------------------------------------------------------------------
template <typename PLAN_HANDLER>
template <typename DERIVED1, typename DERIVED2>
typename local_::enable_if_cc<DERIVED1, DERIVED2>::type
FFTr2c<PLAN_HANDLER>::ft(Eigen::DenseBase<DERIVED1> &dst,
                         const Eigen::DenseBase<DERIVED2> &src,
                         bool scale) const
{
  auto tmp = buf_.get<complex_array_t>(src.rows(), src.cols());
  this->fft2(tmp, src, scale);
  dst.resize(tmp.rows(), tmp.cols());
  fftshift(dst, tmp);
}

// --------------------------------------------------------------------------------
template <typename PLAN_HANDLER>
template <typename DERIVED1, typename DERIVED2>
typename local_::enable_if_cc<DERIVED1, DERIVED2>::type
FFTr2c<PLAN_HANDLER>::ift(Eigen::DenseBase<DERIVED1> &dst,
                          const Eigen::DenseBase<DERIVED2> &src) const
{
  auto tmp = buf_.get<complex_array_t>(src.rows(), src.cols());
  ifftshift(tmp, src);
  this->ifft2(dst, tmp);
}
