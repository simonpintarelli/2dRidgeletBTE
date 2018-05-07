#pragma once

#include <Eigen/Dense>


/**
 * @brief get Fourier modes of array indices
 *        in centered zero frequency convention
 *
 * @param i  array index
 * @param n  array size
 *
 * @return
 */
inline int
to_freq(int i, int n)
{
  int kmax = n / 2;
  int k = i - kmax;
  return k;
}

inline auto
ftgrid(int n)
{
  int o = 1 ? n % 2 == 0 : 0;
  return Eigen::ArrayXd::LinSpaced(n, -n / 2, n / 2 - o);
}
