#pragma once

// system includes ---------------------------------------------------------
#include <malloc.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cassert>
#include <complex>
#include <type_traits>
#include <vector>
// own includes ------------------------------------------------------------
#include "base/types.hpp"
//#include "fft/fft2.hpp"
#include "fft/fft2_r2c.hpp"
#include "fft/ft_grid_helpers.hpp"
// local includes ----------------------------------------------------------
#include "fold.hpp"
#include "init_fftw.hpp"
#include "lambda.hpp"
#include "ridgelet_cell_array.hpp"


namespace internal {

/**
 * @brief dst = src1*src2
 *
 * @param[out] dst
 * @param[in] src1
 * @param[in] src2
 */
template <typename ARRAY, typename DERIVED>
void
mtimes(ARRAY &dst,
       const Eigen::SparseMatrix<double, Eigen::RowMajor> &src1,
       const Eigen::DenseBase<DERIVED> &src2)
{
  typedef Eigen::SparseMatrix<double, Eigen::RowMajor> sp_mat_t;

  assert(src1.rows() == src2.rows());
  assert(src1.cols() == src2.cols());
  assert(dst.rows() == src1.rows());
  assert(dst.cols() == src1.cols());

  dst.setZero();

  for (int k = 0; k < src1.outerSize(); ++k) {
    for (typename sp_mat_t::InnerIterator it(src1, k); it; ++it) {
      int row = it.row();
      int col = it.col();
      double val = it.value();
      dst(row, col) = val * src2(row, col);
    }
  }
}
}  // internal

// forward declaration
class RidgeletFrame;

/**
 * @brief Fast Fourier Ridgelet Transform
 *
 */
template <typename NUMERIC_T = std::complex<double>,
          typename FRAME = RidgeletFrame,
          typename FFT_TYPE = FFTr2c<PlannerR2COD>>
class RT
{
 public:
  typedef Eigen::Array<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      complex_array_t;
  typedef Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> array_t;
  typedef Eigen::Array<NUMERIC_T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> rt_coeff_t;
  typedef FRAME rt_frame_t;
  typedef FFT_TYPE fft_t;
  typedef NUMERIC_T numeric_t;

 public:
  RT(const rt_frame_t &rt_frame)
      : rt_frame_(rt_frame)
  { /* empty */
  }

  RT(const RT<NUMERIC_T, FRAME, FFT_TYPE> &other)
      : rt_frame_(other.rt_frame)
  { /*  empty  */
  }

  RT(RT<NUMERIC_T, FRAME, FFT_TYPE> &&other)
      : rt_frame_(std::move(other.rt_frame))
  { /*  empty  */
  }

  RT<NUMERIC_T, FRAME, FFT_TYPE> &operator=(RT<NUMERIC_T, FRAME, FFT_TYPE> &&other);

  RT<NUMERIC_T, FRAME, FFT_TYPE> &operator=(const RT<NUMERIC_T, FRAME, FFT_TYPE> &other);

  RT() {}

  /**
   * @brief
   *
   * @param[out] dst  Ridgelet coefficients
   * @param[in]  src  Fourier coefficients in _centered zero-frequency_
   * convection
   */
  template <typename DERIVED>
  void rt(std::vector<rt_coeff_t> &dst, const Eigen::DenseBase<DERIVED> &src) const;

  template <typename DERIVED>
  void rt(RidgeletCellArray<rt_coeff_t> &dst, const Eigen::DenseBase<DERIVED> &src) const;

  /**
   * @brief
   *
   * @param[out] dst  Fourier coefficients in _centered zero-frequency_
   * convention
   * @param[in]  src  Ridgelet coefficients (real-valued)
   */
  template <typename DERIVED>
  void irt(Eigen::DenseBase<DERIVED> &dst, const std::vector<rt_coeff_t> &src) const;

  template <typename DERIVED>
  void irt(Eigen::DenseBase<DERIVED> &dst, const RidgeletCellArray<rt_coeff_t> &src) const;

  const rt_frame_t &frame() const { return rt_frame_; }

 private:
  rt_frame_t rt_frame_;
  // buffers
  thread_local static ArrayBuffer<> buf1_;
  thread_local static ArrayBuffer<> buf2_;
};

template <typename NUMERIC_T, typename FRAME, typename FFT_TYPE>
thread_local ArrayBuffer<> RT<NUMERIC_T, FRAME, FFT_TYPE>::buf1_;

template <typename NUMERIC_T, typename FRAME, typename FFT_TYPE>
thread_local ArrayBuffer<> RT<NUMERIC_T, FRAME, FFT_TYPE>::buf2_;

// --------------------------------------------------------------------------------
// --------------------------------------------------------------------------------
template <typename NUMERIC_T, typename FRAME, typename FFT_TYPE>
RT<NUMERIC_T, FRAME, FFT_TYPE> &
RT<NUMERIC_T, FRAME, FFT_TYPE>::operator=(RT<NUMERIC_T, FRAME, FFT_TYPE> &&other)
{
  rt_frame_ = std::move(other.rt_frame_);
}

template <typename NUMERIC_T, typename FRAME, typename FFT_TYPE>
RT<NUMERIC_T, FRAME, FFT_TYPE> &
RT<NUMERIC_T, FRAME, FFT_TYPE>::operator=(const RT<NUMERIC_T, FRAME, FFT_TYPE> &other)
{
  rt_frame_ = other.rt_frame_;
}

template <typename NUMERIC_T, typename FRAME, typename FFT_TYPE>
template <typename DERIVED>
void
RT<NUMERIC_T, FRAME, FFT_TYPE>::rt(std::vector<rt_coeff_t> &dst,
                                   const Eigen::DenseBase<DERIVED> &src) const
{
  static_assert(std::is_same<typename DERIVED::Scalar, complex_array_t::Scalar>::value,
                "type mismatch");
  const auto &lambdas = rt_frame_.lambdas();
  fft_t fft;

  const unsigned int rho_x = rt_frame_.rho_x();
  const unsigned int rho_y = rt_frame_.rho_y();

  for (int i = 0; i < lambdas.size(); ++i) {
    auto &lambda = lambdas[i];

    if (lambda.t == rt_type::S) {
      auto &RF = rt_frame_.get_dense(lambda);
      auto tmp = buf1_.get<complex_array_t>(RF.cols(), RF.rows());
      tmp = ftcut(src, RF.rows(), RF.cols()).array() * RF;
      fft.ift(dst[i], tmp);
    } else {
      // ridgelet coefficients are stored as sparse matrix
      auto &RF = rt_frame_.get_sparse(lambda);
      auto tmp = buf1_.get<complex_array_t>(RF.rows(), RF.cols());
      internal::mtimes(tmp, RF, ftcut(src, RF.rows(), RF.cols()));
      if (lambda.t == rt_type::X) {
        auto tmp2 = buf2_.get<complex_array_t>(8 * rho_y, tmp.cols());
        fold(tmp2, tmp, 8 * rho_y, 0);
        fft.ift(dst[i], tmp2);
      } else if (lambda.t == rt_type::Y) {
        auto tmp2 = buf2_.get<complex_array_t>(tmp.rows(), 8 * rho_x);
        fold(tmp2, tmp, 8 * rho_x, 1);
        fft.ift(dst[i], tmp2);
      } else if (lambda.t == rt_type::D) {
        auto tmp2 = buf2_.get<complex_array_t>(8 * rho_y, tmp.cols());
        fold(tmp2, tmp, 8 * rho_x, 0);
        fft.ift(dst[i], tmp2);
      } else {
        assert(false);
      }
    }
  }
}

// --------------------------------------------------------------------------------
template <typename NUMERIC_T, typename FRAME, typename FFT_TYPE>
template <typename DERIVED>
void
RT<NUMERIC_T, FRAME, FFT_TYPE>::rt(RidgeletCellArray<rt_coeff_t> &dst,
                                   const Eigen::DenseBase<DERIVED> &src) const
{
  this->rt(dst.coeffs(), src);
}
namespace local_ {

template <typename DERIVED, typename T>
void
add_to_dst(Eigen::DenseBase<DERIVED> &dst, const Eigen::Map<T> &out)
{
  int right_nrows = out.rows();
  int right_ncols = out.cols();
  int nrows = dst.rows();
  int ncols = dst.cols();
  int row_offset = nrows / 2 - right_nrows / 2;
  int col_offset = ncols / 2 - right_ncols / 2;
  assert(row_offset >= 0);
  assert(col_offset >= 0);
  dst.block(row_offset, col_offset, right_nrows, right_ncols) += out;
}

}  // local_
// --------------------------------------------------------------------------------
template <typename NUMERIC_T, typename FRAME, typename FFT_TYPE>
template <typename DERIVED>
void
RT<NUMERIC_T, FRAME, FFT_TYPE>::irt(Eigen::DenseBase<DERIVED> &dst,
                                    const std::vector<rt_coeff_t> &src) const
{
  static_assert(std::is_same<typename DERIVED::Scalar, complex_array_t::Scalar>::value,
                "type mismatch");

  assert(dst.rows() == rt_frame_.Ny());
  assert(dst.cols() == rt_frame_.Nx());

  const auto &lambdas = rt_frame_.lambdas();

  dst.setZero();
  fft_t fft;

  for (int i = 0; i < lambdas.size(); ++i) {
    auto &lambda = lambdas[i];
    const bool ft_scale = false;
    auto f_tilde_hat = buf1_.get<complex_array_t>(src[i].rows(), src[i].cols());
    fft.ft(f_tilde_hat, src[i], ft_scale);
    if (lambda.t == rt_type::S) {
      auto &ridge = rt_frame_.get_dense(lambda);
      auto out = buf2_.get<complex_array_t>(ridge.rows(), ridge.cols());
      out = f_tilde_hat * ridge;
      local_::add_to_dst(dst.derived(), out);
    } else if (lambda.t == rt_type::X) {
      auto &ridge = rt_frame_.get_sparse(lambda);
      auto tmp = buf2_.get<complex_array_t>(ridge.rows(), ridge.cols());
      unfold(tmp, f_tilde_hat, ridge.rows(), 0);
      auto out = buf1_.get<complex_array_t>(ridge.rows(), ridge.cols());
      internal::mtimes(out, ridge, tmp);
      local_::add_to_dst(dst.derived(), out);
    } else if (lambda.t == rt_type::Y) {
      auto &ridge = rt_frame_.get_sparse(lambda);
      auto tmp = buf2_.get<complex_array_t>(ridge.rows(), ridge.cols());
      unfold(tmp, f_tilde_hat, ridge.cols(), 1);
      auto out = buf1_.get<complex_array_t>(ridge.rows(), ridge.cols());
      internal::mtimes(out, ridge, tmp);
      local_::add_to_dst(dst.derived(), out);
    } else if (lambda.t == rt_type::D) {
      auto &ridge = rt_frame_.get_sparse(lambda);
      auto tmp = buf2_.get<complex_array_t>(ridge.rows(), ridge.cols());
      unfold(tmp, f_tilde_hat, ridge.rows(), 0);
      auto out = buf1_.get<complex_array_t>(ridge.rows(), ridge.cols());
      internal::mtimes(out, ridge, tmp);
      local_::add_to_dst(dst.derived(), out);
    }
  }
}

// --------------------------------------------------------------------------------
template <typename NUMERIC_T, typename FRAME, typename FFT_TYPE>
template <typename DERIVED>
void
RT<NUMERIC_T, FRAME, FFT_TYPE>::irt(Eigen::DenseBase<DERIVED> &dst,
                                    const RidgeletCellArray<rt_coeff_t> &src) const
{
  this->irt(dst, src.coeffs());
}
