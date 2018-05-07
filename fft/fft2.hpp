#pragma once

#include <fftw3.h>
#include <complex>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <type_traits>

#include "shift.hpp"


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

struct FFT
{
  typedef Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> array_t;
  typedef Eigen::Array<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      complex_array_t;

  typedef std::complex<double> cdouble;

  /**
   * @brief forward transform
   *
   * @param[out] dst
   * @param[in] src
   */
  template <typename DERIVED1, typename DERIVED2>
  typename enable_if_cr<DERIVED1, DERIVED2>::type fft2(Eigen::DenseBase<DERIVED1> &dst,
                                                       const Eigen::DenseBase<DERIVED2> &src,
                                                       bool scale = true) const;

  template <typename DERIVED1, typename DERIVED2>
  typename enable_if_cc<DERIVED1, DERIVED2>::type fft2(Eigen::DenseBase<DERIVED1> &dst,
                                                       const Eigen::DenseBase<DERIVED2> &src,
                                                       bool scale = true) const;

  /**
   * @brief inverse transform (does not preserve input)
   *
   * @param[out] dst
   * @param[in] src   full spectrum
   */
  template <typename DERIVED1, typename DERIVED2>
  typename enable_if_rc<DERIVED1, DERIVED2>::type ifft2(Eigen::DenseBase<DERIVED1> &dst,
                                                        Eigen::DenseBase<DERIVED2> &src) const;

  /**
  * @brief ifft2 (c -> r)
  *
  * @param[out] dst  dest
  * @param[in] src  will be overwritten by FFTW!
  */
  template <typename DERIVED1, typename DERIVED2>
  typename enable_if_cr<DERIVED1, DERIVED2>::type ifft2(Eigen::DenseBase<DERIVED1> &dst,
                                                        Eigen::DenseBase<DERIVED2> &src) const;

  /**
   * @brief ifft2 (c -> c)
   *
   * @param[out] dst  destination
   * @param[in] src  will be overwritten by FFTW!
   */
  template <typename DERIVED1, typename DERIVED2>
  typename enable_if_cc<DERIVED1, DERIVED2>::type ifft2(Eigen::DenseBase<DERIVED1> &dst,
                                                        Eigen::DenseBase<DERIVED2> &src) const;

  /**
   * @brief 2-dim fft (including fftshift)
   *
   * @param[out] dst  complex array (centered zero-frequency convention)
   * @param[in] src   real array
   * @param[in] scale if true: scales output by 1/numel(src)
   *
   * This is the inverse of ift for \var scale set to false.
   *
   */
  template <typename DERIVED1, typename DERIVED2>
  typename enable_if_cr<DERIVED1, DERIVED2>::type ft(Eigen::DenseBase<DERIVED1> &dst,
                                                     const Eigen::DenseBase<DERIVED2> &src,
                                                     bool scale = true) const;

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
  typename enable_if_cc<DERIVED1, DERIVED2>::type ft(Eigen::DenseBase<DERIVED1> &dst,
                                                     const Eigen::DenseBase<DERIVED2> &src,
                                                     bool scale = true) const;

  /**
   * real, complex
   *
   * @brief ifft2 (including ifftshift)
   *
   * @param[out] dst   real array
   * @param[in] src   complex array
   *
   * Scales the output by 1/numel(src). Inverse of ft(dst, src, scale=false).
   *
   */
  template <typename DERIVED1, typename DERIVED2>
  typename enable_if_rc<DERIVED1, DERIVED2>::type ift(Eigen::DenseBase<DERIVED1> &dst,
                                                      const Eigen::DenseBase<DERIVED2> &src) const;

  /**
   * complex, complex
   * @brief complex ifft2 (including ifftshift)
   *
   * @param[out] dst
   * @param[in] src
   *
   * Scales the output by 1/numel(src). Inverse of ft(dst, src, scale=false).
   *
   */
  template <typename DERIVED1, typename DERIVED2>
  typename enable_if_cc<DERIVED1, DERIVED2>::type ift(Eigen::DenseBase<DERIVED1> &dst,
                                                      const Eigen::DenseBase<DERIVED2> &src) const;
};

// --------------------------------------------------------------------------------
template <typename DERIVED1, typename DERIVED2>
typename enable_if_cr<DERIVED1, DERIVED2>::type
FFT::fft2(Eigen::DenseBase<DERIVED1> &dst, const Eigen::DenseBase<DERIVED2> &src, bool scale) const
{
  static_assert(sizeof(fftw_complex) == sizeof(cdouble), "type mismatch");
  // typedef double fftw_cdouble[2];
  typedef fftw_complex fftw_cdouble;
  const int n0 = src.rows();
  const int n1 = src.cols();
  int n[2] = {n0, n1};
  int embed[2] = {n0, n1};
  dst.derived().resize(n0, n1);
  int flags = FFTW_PRESERVE_INPUT | FFTW_ESTIMATE;
  fftw_cdouble *out = reinterpret_cast<fftw_cdouble *>(dst.derived().data());
  double *in = const_cast<double *>(src.derived().data());
  fftw_plan fwd_plan = fftw_plan_many_dft_r2c(2 /* rank */,
                                              n /* dims */,
                                              1 /* num dfts */,
                                              in,
                                              embed,
                                              1 /* stride */,
                                              embed[0] * embed[1],
                                              out,
                                              embed,
                                              1 /*stride */,
                                              embed[0] * embed[1],
                                              flags);
  assert(fwd_plan != NULL);
  fftw_execute(fwd_plan);

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
template <typename DERIVED1, typename DERIVED2>
typename enable_if_cc<DERIVED1, DERIVED2>::type
FFT::fft2(Eigen::DenseBase<DERIVED1> &dst, const Eigen::DenseBase<DERIVED2> &src, bool scale) const
{
  static_assert(sizeof(fftw_complex) == sizeof(cdouble), "type mismatch");

  const int n0 = src.rows();
  const int n1 = src.cols();
  dst.derived().resize(n0, n1);
  fftw_complex *in =
      const_cast<fftw_complex *>(reinterpret_cast<const fftw_complex *>(src.derived().data()));
  fftw_complex *out = reinterpret_cast<fftw_complex *>(dst.derived().data());
  unsigned int flags = FFTW_ESTIMATE | FFTW_PRESERVE_INPUT;
  fftw_plan inv_plan = fftw_plan_dft_2d(n0, n1, in, out, FFTW_FORWARD, flags);

  assert(inv_plan != NULL);
  fftw_execute(inv_plan);
  if (scale) {
    double f = 1. / (n0 * n1);
    dst *= f;
  }
}

// ----------------------------------------------------------------------------
template <typename DERIVED1, typename DERIVED2>
typename enable_if_rc<DERIVED1, DERIVED2>::type
FFT::ifft2(Eigen::DenseBase<DERIVED1> &dst, Eigen::DenseBase<DERIVED2> &src) const
{
  static_assert(sizeof(fftw_complex) == sizeof(cdouble), "type mismatch");
  // typedef double fftw_cdouble[2];
  typedef fftw_complex fftw_cdouble;

  const int n0 = src.rows();
  const int n1 = src.cols();
  int n[2] = {n0, n1};
  dst.derived().resize(n0, n1);
  fftw_cdouble *in = reinterpret_cast<fftw_cdouble *>(src.derived().data());
  double *out = dst.derived().data();
  int embed[2] = {n0, n1};
  unsigned int flags = FFTW_ESTIMATE;
  fftw_plan inv_plan = fftw_plan_many_dft_c2r(
      2, n, 1, in, embed, 1, embed[0] * embed[1], out, embed, 1, embed[0] * embed[1], flags);
  assert(inv_plan != NULL);
  fftw_execute(inv_plan);

  double f = 1. / (n0 * n1);
  dst *= f;
}

// --------------------------------------------------------------------------------
template <typename DERIVED1, typename DERIVED2>
typename enable_if_cc<DERIVED1, DERIVED2>::type
FFT::ifft2(Eigen::DenseBase<DERIVED1> &dst, Eigen::DenseBase<DERIVED2> &src) const
{
  const int n0 = src.rows();
  const int n1 = src.cols();

  dst.derived().resize(n0, n1);

  fftw_complex *in = reinterpret_cast<fftw_complex *>(src.derived().data());
  fftw_complex *out = reinterpret_cast<fftw_complex *>(dst.derived().data());
  unsigned int flags = FFTW_ESTIMATE;
  // fftw_plan fftw_plan_dft_2d(int n0, int n1,
  //                          fftw_complex *in, fftw_complex *out,
  //                          int sign, unsigned flags);
  fftw_plan inv_plan = fftw_plan_dft_2d(n0, n1, in, out, FFTW_BACKWARD, flags);

  assert(inv_plan != NULL);
  fftw_execute(inv_plan);
  double f = 1. / (n0 * n1);
  dst *= f;
}

// --------------------------------------------------------------------------------
template <typename DERIVED1, typename DERIVED2>
typename enable_if_cr<DERIVED1, DERIVED2>::type
FFT::ifft2(Eigen::DenseBase<DERIVED1> &dst, Eigen::DenseBase<DERIVED2> &src) const
{
  // not implemented
  throw 1;
}

// --------------------------------------------------------------------------------
template <typename DERIVED1, typename DERIVED2>
typename enable_if_cr<DERIVED1, DERIVED2>::type
FFT::ft(Eigen::DenseBase<DERIVED1> &dst, const Eigen::DenseBase<DERIVED2> &src, bool scale) const
{
  complex_array_t tmp(src.rows(), src.cols());
  this->fft2(tmp, src, scale);
  dst.derived().resize(tmp.rows(), tmp.cols());
  fftshift(dst, tmp);
}

// --------------------------------------------------------------------------------
template <typename DERIVED1, typename DERIVED2>
typename enable_if_cc<DERIVED1, DERIVED2>::type
FFT::ft(Eigen::DenseBase<DERIVED1> &dst, const Eigen::DenseBase<DERIVED2> &src, bool scale) const
{
  complex_array_t tmp(src.rows(), src.cols());
  this->fft2(tmp, src, scale);
  dst.derived().resize(tmp.rows(), tmp.cols());
  fftshift(dst, tmp);
}

// --------------------------------------------------------------------------------
template <typename DERIVED1, typename DERIVED2>
typename enable_if_rc<DERIVED1, DERIVED2>::type
FFT::ift(Eigen::DenseBase<DERIVED1> &dst, const Eigen::DenseBase<DERIVED2> &src) const
{
  complex_array_t tmp(src.rows(), src.cols());
  ifftshift(tmp, src);
  this->ifft2(dst, tmp);
}

// --------------------------------------------------------------------------------
template <typename DERIVED1, typename DERIVED2>
typename enable_if_cc<DERIVED1, DERIVED2>::type
FFT::ift(Eigen::DenseBase<DERIVED1> &dst, const Eigen::DenseBase<DERIVED2> &src) const
{
  complex_array_t tmp(src.rows(), src.cols());
  ifftshift(tmp, src);
  this->ifft2(dst, tmp);
}
