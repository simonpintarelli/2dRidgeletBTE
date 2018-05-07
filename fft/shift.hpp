#pragma once

#include <Eigen/Dense>
#include <cmath>


/**
 * @brief shift zero frequency component to the center of the spectrum
 *
 * @param dst
 * @param src
 * @param dim (default both directions), 0: shift along rows, 1: shift along
 * columns
 */
template <typename DERIVED1, typename DERIVED2>
inline void
fftshift(Eigen::DenseBase<DERIVED1> &dst, const Eigen::DenseBase<DERIVED2> &src, int dim = -1)
{
  // nx, ny: position of the smallest negative frequency
  unsigned int nx = std::ceil(src.cols() / 2.);
  unsigned int ny = std::ceil(src.rows() / 2.);

  unsigned int Nx = src.cols();
  unsigned int Ny = src.rows();

  assert(dst.rows() == src.rows() && dst.cols() == src.cols());

  if (dim == -1 && dst.cols() > 1 && dst.rows() > 1) {
    // diagonal blocks
    dst.block(0, 0, Ny - ny, Nx - nx) = src.bottomRightCorner(Ny - ny, Nx - nx);
    dst.bottomRightCorner(ny, nx) = src.block(0, 0, ny, nx);

    // off-diagonal blocks
    dst.topRightCorner(Ny - ny, nx) = src.bottomLeftCorner(Ny - ny, nx);
    dst.bottomLeftCorner(ny, Nx - nx) = src.topRightCorner(ny, Nx - nx);
  } else if (dim == 0) {
    assert(Ny > 1);
    /*  shift along rows  */
    // 2d, shift along rows (y-dim)
    dst.topRows(Ny - ny) = src.bottomRows(Ny - ny);
    dst.bottomRows(ny) = src.topRows(ny);
  } else if (dim == 1) {
    assert(Nx > 1);
    /* shift along columns  */
    // 2d, shift along columns (x-dim)
    // negative frequencies
    dst.leftCols(Nx - nx) = src.rightCols(Nx - nx);
    // positive frequencies
    dst.rightCols(nx) = src.leftCols(nx);
  } else {
    assert(false);
  }
}

/**
 * @brief undoes the effect of fftshift
 *
 * @param dst
 * @param src
 * @param dim
 */
template <typename DERIVED1, typename DERIVED2>
inline void
ifftshift(Eigen::DenseBase<DERIVED1> &dst, const Eigen::DenseBase<DERIVED2> &src, int dim = -1)
{
  // nx, ny: position of the smallest negative frequency
  unsigned int nx = std::floor(src.cols() / 2.);
  unsigned int ny = std::floor(src.rows() / 2.);

  unsigned int Nx = src.cols();
  unsigned int Ny = src.rows();

  assert(dst.rows() == src.rows() && dst.cols() == src.cols());

  if (dim == -1 && dst.cols() > 1 && dst.rows() > 1) {
    // diagonal blocks
    dst.block(0, 0, Ny - ny, Nx - nx) = src.bottomRightCorner(Ny - ny, Nx - nx);
    dst.bottomRightCorner(ny, nx) = src.block(0, 0, ny, nx);

    // off-diagonal blocks
    dst.topRightCorner(Ny - ny, nx) = src.bottomLeftCorner(Ny - ny, nx);
    dst.bottomLeftCorner(ny, Nx - nx) = src.topRightCorner(ny, Nx - nx);
  } else if (dim == 0) {
    // 2d, shift along rows (y-dim)
    dst.topRows(Ny - ny) = src.bottomRows(Ny - ny);
    dst.bottomRows(ny) = src.topRows(ny);
  } else if (dim == 1) {
    // 2d, shift along columns (x-dim)
    // negative frequencies
    dst.leftCols(Nx - nx) = src.rightCols(Nx - nx);
    // positive frequencies
    dst.rightCols(nx) = src.leftCols(nx);
  }
}
