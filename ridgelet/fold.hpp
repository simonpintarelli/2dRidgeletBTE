#pragma once

#include <Eigen/Dense>
#include "base/array_buffer.hpp"
#include "fft/shift.hpp"

/**
 * @brief fold operation,
 *
 *
 * @param[out] dst (resized)
 *  dim == 0: width x m matrix
 *  dim == 1: n x width matrix
 * @param[in]  src   Fourier coefficients of f(x), matrix of size n x m
 * @param[in]  width
 * @param[in]  dim   dimension along which to fold
 *
 */
template <typename DERIVED1, typename DERIVED2>
void
fold(Eigen::DenseBase<DERIVED1> &dst,
     const Eigen::DenseBase<DERIVED2> &src,
     unsigned int width,
     int dim)
{
  const int ncols = src.cols();
  const int nrows = src.rows();

  typedef typename DERIVED1::Scalar numeric_t;
  typedef Eigen::Array<numeric_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> local_array_t;

  thread_local static ArrayBuffer<> buf;

  if (dim == 0) {
    assert(nrows % width == 0);
    if ((src.rows() / width) % 2 == 0) {
      const int s = nrows / width;
      auto tmp = buf.get<local_array_t>(width, ncols);
      tmp.setZero();
      for (int i = 0; i < (int)width; ++i) {
        for (int k = 0; k < s; ++k) {
          tmp.row(i) += src.row(i + k * width);
        }
      }
      // apply fftshift along dim
      fftshift(dst, tmp, dim);
    } else {
      const int s = nrows / width;
      assert(dst.rows() == width);
      assert(dst.cols() == ncols);
      dst.setZero();
      for (int i = 0; i < (int)width; ++i) {
        for (int k = 0; k < s; ++k) {
          dst.row(i) += src.row(i + k * width);
        }
      }
    }
  } else if (dim == 1) {
    assert(ncols % width == 0);
    if ((src.cols() / width) % 2 == 0) {
      const int s = ncols / width;
      auto tmp = buf.get<local_array_t>(nrows, width);
      tmp.setZero();
      for (int j = 0; j < (int)width; ++j) {
        for (int k = 0; k < s; ++k) {
          tmp.col(j) += src.col(j + k * width);
        }
      }
      // apply fftshift along dim
      fftshift(dst, tmp, dim);
    } else {
      const int s = ncols / width;
      assert(dst.rows() == nrows);
      assert(dst.cols() == width);
      dst.setZero();
      for (int j = 0; j < (int)width; ++j) {
        for (int k = 0; k < s; ++k) {
          dst.col(j) += src.col(j + k * width);
        }
      }
    }
  }
}

/**
 * @brief unfold operation.
 *
 * unfold scales src by 1/k where k := width/size(src,dim) and repeats it
 * along dimension dim such that size(src_unfolded, dim) = width.
 * If dim == 0, the rows of src_unfolded are shifted such that the middle row of
 * src
 * becomes the middle row of src_unfolded, where the middle is determined
 * according to the centered-zero-frequency convention. In this case, this
 * function is mathematically equivalent to
 *   temp = zeros(width, size(src,2));
 *   temp(1:width/size(src,1):end, :) = ift(src);
 *   dst = ft(temp);
 * Likewise properties hold for dim == 1.
 *
 *
 * @param[out] dst  (resized)
 *   dim == 0: dst = width x m
 *   dim == 1: dst = n x width
 * @param[in] src of size n x m
 * @param[in] width
 * @param[in] dim
 * @param[in] scale
 */
template <typename DERIVED1, typename DERIVED2>
void
unfold(Eigen::DenseBase<DERIVED1> &dst,
       Eigen::DenseBase<DERIVED2> &src,
       unsigned int width,
       int dim,
       bool scale = false)
{
  typedef typename DERIVED1::Scalar numeric_t;
  typedef Eigen::Array<numeric_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> local_array_t;

  thread_local static ArrayBuffer<> buf1;
  thread_local static ArrayBuffer<> buf2;

  assert(dim == 0 || dim == 1);
  const int nrows = src.rows();
  const int ncols = src.cols();

  if (dim == 0) {
    const int s = width / nrows;
    assert(s > 0);
    if ((width / nrows) % 2 == 0) {
      auto tmp = buf1.get<local_array_t>(src.rows(), src.cols());
      ifftshift(tmp, src, 0);
      auto tmp2 = buf2.get<local_array_t>(width, src.cols());
      tmp2 = tmp.replicate(s, 1);
      assert(dst.rows() == width);
      fftshift(dst, tmp2, 0);
    } else {
      dst.derived() = src.replicate(s, 1);
    }
    if (scale) dst *= src.rows() / double(width);
  } else if (dim == 1) {
    const int s = width / ncols;
    assert(s > 0);
    if (s % 2 == 0) {
      auto tmp = buf1.get<local_array_t>(src.rows(), src.cols());
      ifftshift(tmp, src, 1);
      auto tmp2 = buf2.get<local_array_t>(src.rows(), width);
      tmp2 = tmp.replicate(1, s);
      assert(dst.cols() == width);
      fftshift(dst, tmp2, 1);
    } else {
      dst = src.replicate(1, s);
    }
    if (scale) dst *= src.cols() / double(width);
  }
}
