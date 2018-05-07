#pragma once

#include <Eigen/Sparse>


/**
 *
 *
 * @param ft  Sparse n x m matrix in centered zero frequency convention
 * @param nx  number of columns!
 * @param ny  number of rows!
 *
 * @return sparse matrix of size (2 ny) x (2 nx), with frequencies
 * (-nx,..,0,...nx-1) and
 * (-ny,... ,0, ...ny-1) respectively in centered zero frequency convention
 *
 * @return
 */
template <NUMERIC_T, int _OPTIONS, int _INDEX>
Eigen::Block<Eigen::SparseMatrix<NUMERIC_T, _OPTIONS, _INDEX>, Eigen::Dynamic, Eigen::Dynamic>
ftcut(const Eigen::SparseMatrix<NUMERIC_T, _OPTIONS, _INDEX> &ft, int nx, int ny)
{
  assert(ft.rows() % 2 == 0);
  assert(ft.cols() % 2 == 0);
  unsigned int nx_old = ft.rows() / 2;
  unsigned int ny_old = ft.cols() / 2;

  assert(ft.rows() >= 2 * ny);
  assert(ft.cols() >= 2 * nx);
  return ft.block(-ny + ny_old, -nx + nx_old, 2 * ny, 2 * nx);
}
