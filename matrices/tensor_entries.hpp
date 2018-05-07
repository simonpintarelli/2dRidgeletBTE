#pragma once

#include <Eigen/Dense>
#include <boost/assert.hpp>
#include <boost/math/constants/constants.hpp>

#include <iostream>


const double PI = boost::math::constants::pi<double>();

template <typename FT2, typename FT3>
double
overlap3_inner(const FT2 &ft2,
               const FT3 &ft3,
               int k1,
               int k2,
               double txp,
               double txpp,
               double tx,
               double typ,
               double typp,
               double ty)
{
  // k' <=> ft2
  // k'' <=> ft3
  int Np = ft2.rows();
  int Mp = ft2.cols();
  int Npp = ft3.rows();
  int Mpp = ft3.cols();

  BOOST_VERIFY(Np % 2 == 0);
  BOOST_VERIFY(Mp % 2 == 0);
  BOOST_VERIFY(Npp % 2 == 0);
  BOOST_VERIFY(Mpp % 2 == 0);

  int np = Np / 2;
  int mp = Mp / 2;
  int npp = Npp / 2;
  int mpp = Mpp / 2;

  double vsum = 0;

  for (int k1p = std::max(-npp + 1 - k1, -np); k1p <= std::min(npp - k1, np - 1); ++k1p) {
    int i1p = k1p + np;
    for (typename FT2::InnerIterator itp(ft2, i1p); itp; ++itp) {
      int i2p = itp.col();
      int k2p = i2p - mp;
      int k1pp = -k1 - k1p;
      int i1pp = k1pp + npp;
      int k2pp = -k2 - k2p;
      int i2pp = k2pp + mpp;

      if (k2pp >= -mpp + 1 && k2pp <= mpp - 1)
        vsum += itp.value() * ft3.coeff(i1pp, i2pp) *
                std::cos(2 * PI *
                         (k2p * typ + k2pp * typp + k2 * ty + k1p * txp + k1pp * txpp + k1 * tx));
    }
  }
  // std::cout << "hits: (" << k1 << ", " << k2 << "): " << hits << "\n";
  return vsum;
}

/**
 *
 *
 * @param ft1
 * @param ft2
 * @param ft3
 * @param tp    \f$t' \in [0,1] \f$
 * @param tpp   \f$t'' \in [0,1] \f$
 * @param t     \f$t \in [0,1] \f$
 *
 * @return
 */
template <typename FT1, typename FT2, typename FT3>
double
overlap3(const FT1 &ft1,
         const FT2 &ft2,
         const FT3 &ft3,
         const Eigen::Vector2d &tp = {0, 0},
         const Eigen::Vector2d &tpp = {0, 0},
         const Eigen::Vector2d &t = {0, 0})
{
  int N = ft1.rows();
  int n = N / 2;
  int M = ft1.cols();
  int m = M / 2;

  BOOST_VERIFY(N % 2 == 0);
  BOOST_VERIFY(M % 2 == 0);

  double txp = tp[0];
  double typ = tp[1];
  double txpp = tpp[0];
  double typp = tpp[1];
  double tx = t[0];
  double ty = t[1];
  double vsum = 0;
  for (int i = 0; i < N; ++i) {
    int k1 = i - n;
    if (k1 >= 0)
      for (typename FT1::InnerIterator it(ft1, i); it; ++it) {
        int j = it.col();
        int k2 = j - m;
        double f1 = it.value();
        if (k1 == 0)
          vsum += f1 * overlap3_inner(ft2, ft3, k1, k2, txp, txpp, tx, typ, typp, ty);
        else
          vsum += 2 * f1 * overlap3_inner(ft2, ft3, k1, k2, txp, txpp, tx, typ, typp, ty);
      }
    else
      continue;
  }
  return vsum;
}

// center ft1 on (0, 0) frequency at (ky, kx) in ft2 and compute cwise prod and
// sum
template <typename FT>
double
overlap_simple_inner(const FT &ft1,
                     const FT &ft2,
                     int ky,  // row frequency
                     int kx,  // col frequency
                     const Eigen::Vector2d &t12,
                     const Eigen::Vector2d &t23)
{
  // std::cout << "overlap_simple_inner (ky, kx): " << ky << " " << kx << "\n";
  int N1 = ft1.rows();
  int M1 = ft1.cols();

  int N2 = ft2.rows();
  int M2 = ft2.cols();

  static_assert(FT::IsRowMajor);

  // assume n, m even!
  assert(N1 % 2 == 0);
  assert(M1 % 2 == 0);
  assert(N2 % 2 == 0);
  assert(M2 % 2 == 0);

  int n1 = N1 / 2;
  int m1 = M1 / 2;
  int n2 = N2 / 2;
  int m2 = M2 / 2;

  // freq. range of ft1
  int y1h_min = -n1;
  int y1h_max = n1 - 1;
  int x1h_min = -m1;
  int x1h_max = m1 - 2;

  // freq. range of ft2
  int y2h_min = -n2;
  int y2h_max = n2 - 1;
  int x2h_min = -m2;
  int x2h_max = m2 - 1;

  // compute the minimum frequency range of ft1 that has to be considered
  int y1h_beg = std::max(-n1, 1 - n2 + ky);
  int y1h_end = std::min(n1 - 1, n2 + ky);
  int x1h_beg = std::max(-m1, 1 - m2 + kx);
  int x1h_end = std::min(m1 - 1, m2 + kx);

  // define function which maps frequencies (yhat, xhat) to array indices (i,
  // j).
  auto to_i1 = [n1](int y1h) { return y1h + n1; };
  auto to_j1 = [m1](int x1h) { return x1h + m1; };
  auto to_i2 = [n2](int y2h) { return y2h + n2; };
  auto to_j2 = [m2](int x2h) { return x2h + m2; };
  // and their inverses:
  auto to_y1 = [n1](int i1) { return i1 - n1; };
  auto to_x1 = [m1](int j1) { return j1 - m1; };
  auto to_y2 = [n2](int i2) { return i2 - n2; };
  auto to_x2 = [m2](int j2) { return j2 - m2; };

  // std::cout << "y in [" << y1h_beg << ", " << y1h_end << "]"
  //           << std::endl
  //           << "x in [" << x1h_beg << ", " << x1h_end << "]"
  //           << std::endl;

  const Eigen::Vector2d xh = {ky, kx};

  typedef typename FT::InnerIterator it_t;
  typedef typename FT::StorageIndex StorageIndex;
  double vsum = 0;

  const StorageIndex *ft2_cols = ft2.innerIndexPtr();
  const StorageIndex *ft2_outer = ft2.outerIndexPtr();
  const double *ft2_values = ft2.valuePtr();

  // iterate over rows in ft1
  for (int y1h = y1h_beg; y1h <= y1h_end; ++y1h) {
    // obtain row of ft1
    it_t ity1(ft1, to_i1(y1h));
    int x1h = to_x1(ity1.col());

    // std::cout << "x1h (first nnz): " << x1h << "\t";
    // std::cout << "xbegin: " << x1h_beg << "\n";

    // search forward in ft1
    while (to_x1(ity1.col()) < x1h_beg) {
      // std::cout << " nnz... " << to_x1(ity1.col()) << "\n";
      ++ity1;
    }
    if (!bool(ity1)) {
      continue;  // skip, there are no entries in ft1(y1h, :) to consider
    }

    int y2h = ky - y1h;  // corresp. row freq. in ft2
    // get raw Eigen pointers to current row in ft2
    int ft2_col = to_i2(y2h);
    int ft2_row_begin = ft2_outer[ft2_col];
    int ft2_row_end = ft2_outer[ft2_col + 1];

    // get last nonzero position in current row of ft2
    int ft2_rindex = ft2_row_end - 1;
    // iterate over columns in ft1
    for (; bool(ity1) && to_x1(ity1) <= x1h_end; ++ity1) {
      x1h = to_x1(ity1.col());
      // search backward in ft2
      // Note: to_x2(ft2_cols[ft2_rindex]) => x2h
      while (to_x2(ft2_cols[ft2_rindex]) > kx - x1h && ft2_rindex >= ft2_row_begin) --ft2_rindex;
      int x2h = to_x2(ft2_cols[ft2_rindex]);
      if (x2h == kx - x1h) {
        // matching position found, update
        double ft2_value = ft2_values[ft2_rindex];
        const Eigen::Vector2d xhp = {y1h, x1h};
        vsum += std::cos(2 * PI * t12.dot(xhp) + 2 * PI * t23.dot(xh)) * ity1.value() * ft2_value;
      } else if (x2h < kx - x1h_end) {
        // x2h hast left the range of ft1,
        // done with current row, continue with next row (y1h).
        break;
      }
    }
  }
  return vsum;
}

template <typename FT>
double
overlap3_simple(const FT &ft1,
                const FT &ft2,
                const FT &ft3,
                const Eigen::Vector2d &t12 = {0, 0},
                const Eigen::Vector2d &t23 = {0, 0})
{
  int N = ft1.rows();
  int M = ft1.cols();

  assert(N % 2 == 0);
  assert(M % 2 == 0);

  int n = N / 2;
  int m = M / 2;
  double vsum = 0;

  for (int i = n; i < N; ++i) {
    int k1 = i - n;
    // std::cout << "overlap3_simple::k1 " << k1 << "\n";

    if (k1 >= 0)
      for (typename FT::InnerIterator it(ft1, i); it; ++it) {
        int j = it.col();
        int k2 = j - m;
        double f1 = it.value();

        if (k1 == 0)
          vsum += f1 * overlap_simple_inner(ft2, ft3, k1, k2, t12, t23);
        else
          vsum += 2 * f1 * overlap_simple_inner(ft2, ft3, k1, k2, t12, t23);
      }
    else
      continue;
  }
  return vsum;
}
