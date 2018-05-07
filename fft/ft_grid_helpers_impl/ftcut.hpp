#pragma once

#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <boost/assert.hpp>

#include <fstream>
#include <iostream>


template <typename T>
Eigen::Block<const T>
ftcut(const Eigen::DenseBase<T> &m, int rows, int cols)
{
  int this_rows = m.rows();
  int this_cols = m.cols();

  int row_offset = this_rows / 2 - rows / 2;
  int col_offset = this_cols / 2 - cols / 2;
  assert(row_offset >= 0);
  assert(col_offset >= 0);
  return m.block(row_offset, col_offset, rows, cols);
}

template <typename T>
Eigen::Block<T>
ftcut(Eigen::DenseBase<T> &m, int rows, int cols)
{
  int this_rows = m.rows();
  int this_cols = m.cols();

  int row_offset = this_rows / 2 - rows / 2;
  int col_offset = this_cols / 2 - cols / 2;
  assert(row_offset >= 0);
  assert(col_offset >= 0);
  return m.block(row_offset, col_offset, rows, cols);
}

template <typename T>
Eigen::Block<const T>
ftcut(const Eigen::SparseMatrixBase<T> &m, int rows, int cols)
{
  int this_rows = m.rows();
  int this_cols = m.cols();

  int row_offset = this_rows / 2 - rows / 2;
  int col_offset = this_cols / 2 - cols / 2;
  assert(row_offset >= 0);
  assert(col_offset >= 0);
  return m.block(row_offset, col_offset, rows, cols);
}

template <typename T>
Eigen::Block<T>
ftcut(Eigen::DenseBase<T> &&m, int rows, int cols)
{
  (void)m;
  (void)rows;
  (void)cols;
  static_assert(sizeof(T) == -1, "invalid instantiation");
}

template <typename DERIVED1, typename DERIVED2>
void
ftpad(Eigen::SparseMatrixBase<DERIVED1> &dst,
      const Eigen::SparseMatrixBase<DERIVED2> &src,
      int nrows,
      int ncols)
{
  BOOST_VERIFY(nrows >= src.rows() && ncols >= src.cols());

  // new row and column offsets
  auto offset = [](int nnew, int nold) {
    if ((nnew % 2) == (nold % 2)) {
      /*  (even,even) or (odd, odd) */
      return (nnew - nold) / 2;
    } else if (nnew % 2 == 0 && nold % 2 == 1) {
      /*  (even, odd)  */
      return (nnew - nold + 1) / 2;
    } else if (nnew % 2 == 1 && nold % 2 == 0) {
      /* (odd, even)  */
      return (nnew - nold) / 2;
    } else {
      assert(false);
    }
  };

  int row_offset = offset(nrows, src.rows());
  int col_offset = offset(ncols, src.cols());

  // todo: this could be implemented (on-the-fly) as a transformation returning
  // an iterator
  std::vector<Eigen::Triplet<double, int>> triplets(src.derived().nonZeros());
  int count = 0;
  for (int ii = 0; ii < src.outerSize(); ++ii) {
    for (typename DERIVED2::InnerIterator it(src.derived(), ii); it; ++it) {
      triplets[count++] =
          Eigen::Triplet<double, int>(it.row() + row_offset, it.col() + col_offset, it.value());
    }
  }
  BOOST_VERIFY(count == src.derived().nonZeros());
  dst.derived().resize(nrows, ncols);
  dst.derived().setFromTriplets(triplets.begin(), triplets.end());
}

template <typename DERIVED1, typename DERIVED2>
void
ftpad(Eigen::SparseMatrixBase<DERIVED2> &src, int nrows, int ncols)
{
  BOOST_VERIFY(nrows >= src.rows() && ncols >= src.cols());

  // new row and column offsets
  auto offset = [](int nnew, int nold) {
    if ((nnew % 2) == (nold % 2)) {
      /*  (even,even) or (odd, odd) */
      return (nnew - nold) / 2;
    } else if (nnew % 2 == 0 && nold % 2 == 1) {
      /*  (even, odd)  */
      return (nnew - nold + 1) / 2;
    } else if (nnew % 2 == 1 && nold % 2 == 0) {
      /* (odd, even)  */
      return (nnew - nold) / 2;
    } else {
      assert(false);
    }
  };

  int row_offset = offset(nrows, src.rows());
  int col_offset = offset(ncols, src.cols());

  // todo: this could be implemented (on-the-fly) as a transformation returning
  // an iterator
  std::vector<Eigen::Triplet<double, int>> triplets(src.nonZeros());
  int count = 0;
  for (int ii = 0; ii < src.outerSize(); ++ii) {
    for (typename DERIVED2::InnerIterator it(src.derived(), ii); it; ++it) {
      triplets[count++] =
          Eigen::Triplet<double, int>(it.row() + row_offset, it.col() + col_offset, it.value());
    }
  }
  BOOST_VERIFY(count == src.derived().nonZeros());
  src.derived().resize(nrows, ncols);
  src.derived().setFromTriplets(triplets.begin(), triplets.end());
}

/**
 *  @brief set maximal negative frequency coefficients to zero
 *
 *
 *  @param Y Fourier coefficients in zero-centered frequency storage
 */
template <typename DERIVED>
void
hf_zero(Eigen::DenseBase<DERIVED> &Y)
{
  Y.derived().col(0).setZero();
  Y.derived().row(0).setZero();
}
