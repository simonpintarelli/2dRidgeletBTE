#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <limits>
#include "base/exceptions.hpp"
#include "base/pow2p.hpp"
#include "ridgelet_functions.hpp"

template <template <typename> class PSI_RADIAL = PsiRadial1,
          template <typename> class PSI_SPHERICAL = PsiSpherical1,
          template <typename> class PSI_SCALING = PsiScaling1,
          typename TRANSITION_FUNCTION = TransitionFunction,
          typename NUMERIC = double>
class ridge_ft
{
 public:
  typedef NUMERIC numeric_t;
  typedef Eigen::SparseMatrix<numeric_t, Eigen::RowMajor> sparse_mat_t;
  typedef Eigen::Array<numeric_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> mat_t;

 protected:
  typedef TRANSITION_FUNCTION tfct_t;
  typedef PSI_RADIAL<tfct_t> psi_radial_t;
  typedef PSI_SPHERICAL<tfct_t> psi_spherical_t;
  typedef PSI_SCALING<tfct_t> psi_scaling_t;

 public:
  static void d_ridgelet_ft(sparse_mat_t &ft,
                            const sparse_mat_t &ft_x,
                            const sparse_mat_t &ft_y,
                            unsigned int j,
                            unsigned int rho_x,
                            unsigned int rho_y);

  static void x_ridgelet_ft(
      sparse_mat_t &ft, unsigned int j, int k, unsigned int rho_x, unsigned int rho_y);

  static void y_ridgelet_ft(
      sparse_mat_t &ft, unsigned int j, int k, unsigned int rho_x, unsigned int rho_y);

  static void s_ridgelet_ft(mat_t &ft, unsigned int rho_x, unsigned int rho_y);
};

// --------------------------------------------------------------------------------
/**
 * @brief
 *
 * @param ft
 * @param ft_x   \f$\Psi_{j,x,k}\f$
 * @param ft_y   \f$\Psi_{j,y,k}\f$
 * @param rho_x
 * @param rho_y
 */
template <template <class> class PSI_RADIAL,
          template <class> class PSI_SPHERICAL,
          template <class> class PSI_SCALING,
          typename TRANSITION_FUNCTION,
          typename NUMERIC>
void
ridge_ft<PSI_RADIAL, PSI_SPHERICAL, PSI_SCALING, TRANSITION_FUNCTION, NUMERIC>::d_ridgelet_ft(
    sparse_mat_t &ft,
    const sparse_mat_t &ft_x,
    const sparse_mat_t &ft_y,
    unsigned int j,
    unsigned int rho_x,
    unsigned int rho_y)
{
  auto freq = [](int x, int nx) { return x - nx; };
  // output size

  double rhoyox = double(rho_y) / rho_x;
  unsigned int new_rows = pow2p(j + 2) * rho_y;
  unsigned int new_cols = pow2p(j + 2) * rho_x;
  int fcutx = new_cols / 2;
  int fcuty = new_rows / 2;

  ft.resize(new_rows, new_cols);

  // nrows <-> y
  // ncols <-> x
  typedef Eigen::Triplet<double, int> T;
  std::vector<T> entries;
  entries.reserve(ft_x.nonZeros());

  unsigned int nrowsh = ft_x.rows() / 2;
  unsigned int ncolsh = ft_x.cols() / 2;
  for (int ii = 0; ii < ft_x.outerSize(); ++ii) {
    for (typename sparse_mat_t::InnerIterator it(ft_x, ii); it; ++it) {
      int y = freq(it.row(), nrowsh);
      int x = freq(it.col(), ncolsh);
      int posx = x + fcutx;
      int posy = y + fcuty;
      int ax = std::abs(x);
      int ay = std::abs(y);
      if (rhoyox * ax >= ay && posx < new_cols && posx >= 0 && y + fcuty < new_rows && posy >= 0)
        entries.push_back(T(posy, posx, it.value()));
    }
  }

  nrowsh = ft_y.rows() / 2;
  ncolsh = ft_y.cols() / 2;
  for (int ii = 0; ii < ft_y.outerSize(); ++ii) {
    for (typename sparse_mat_t::InnerIterator it(ft_y, ii); it; ++it) {
      int y = freq(it.row(), nrowsh);
      int x = freq(it.col(), ncolsh);
      int posx = x + fcutx;
      int posy = y + fcuty;
      int ax = std::abs(x);
      int ay = std::abs(y);
      if (rhoyox * ax < ay && posx < new_cols && posx >= 0 && y + fcuty < new_rows && posy >= 0)
        entries.push_back(T(y + fcuty, x + fcutx, it.value()));
    }
  }

  // duplicates are *SUMMED* up
  ft.setFromTriplets(entries.begin(), entries.end());
  ft.makeCompressed();
}

// --------------------------------------------------------------------------------
template <template <class> class PSI_RADIAL,
          template <class> class PSI_SPHERICAL,
          template <class> class PSI_SCALING,
          typename TRANSITION_FUNCTION,
          typename NUMERIC>
void
ridge_ft<PSI_RADIAL, PSI_SPHERICAL, PSI_SCALING, TRANSITION_FUNCTION, NUMERIC>::x_ridgelet_ft(
    sparse_mat_t &ft, unsigned int j, int k, unsigned int rho_x, unsigned int rho_y)
{
  psi_radial_t psi_radial;
  psi_spherical_t psi_spherical;

  assert(j > 0);
  assert(k == 0 || std::abs(k) <= pow2p(j - 1));

  typedef Eigen::VectorXd vec_t;
  std::vector<unsigned int> vecd_t;
  typedef Eigen::Triplet<numeric_t, int> T;

  // rho_y/float(rho_x)/2**(j-1)*(1+abs(k))*xmax
  int Nrows = 4 * rho_y * (std::abs(k) + 1);
  int Ncols = pow2p(j + 1) * rho_x;

  // upper bound for number of nonzero elements found in  {(x,y) : x > 0}
  // 2*2**(-1-j)*(float(rho_y)/rho_x)*(2+3*2**j*rho_x)*(5*2**j*rho_x-4)
  // unsigned int nnz = 1./pow2p(j+1) * double(rho_y) / rho_x * (2 +
  // 3*pow2p(j)*rho_x) *
  // (5*pow2p(j)*rho_x-4);
  const unsigned int nnz = 5 * (2 + 3 * pow2p(j) * rho_x) * rho_y;
  std::vector<T> triplets;
  triplets.reserve(2 * nnz);
  ft.resize(2 * Nrows, 2 * Ncols);

  /// initialize helper
  // x' \in [2^(j+1) rho_x,..., 2^(j+1) rho_x]
  const unsigned int x_begin = pow2p(j - 1) * rho_x;
  //  std::cout << "xbegin: " << x_begin << "\n";
  const unsigned int x_end = pow2p(j + 1) * rho_x - 1;
  //  std::cout << "xend: " << x_end << "\n";
  const unsigned int nx = x_end - x_begin + 1;
  std::vector<double> wr(nx);
  unsigned int n = x_begin;
  const double d = (1. / pow2p(j - 1)) / rho_x;
  std::generate(wr.begin(), wr.end(), [&n, &psi_radial, d]() { return psi_radial((n++) * d); });

  unsigned int count = 0;

  const double c1p2_jm1 = 1. / pow2p(j - 1);
  const double rho_yox = rho_y / double(rho_x);
  /// assemble
  for (int x = x_begin, i = 0; x <= x_end; ++x, ++i) {
    // x coordinate on reference frame
    // int xr = x>>(j-1);
    int yl = (k - 1) * rho_yox * c1p2_jm1 * x;
    int yh = (k + 1) * rho_yox * c1p2_jm1 * x;
    //    std::cout << "x: " << x << ", y in " << yl << ".." << yh << "\n";
    for (int y = yl; y <= yh; ++y) {
      // -((k rx)/ry)+(2^(-1+j) rx y)/(ry x)
      double z = rho_x / double(rho_y) * pow2p(j - 1) * y / double(x) - k;
      double ws = psi_spherical(z);
      double val = wr[i] * ws;
      // double val = 1;

      // debug
      assert(y + Nrows < 2 * Nrows);
      assert(-y + Nrows < 2 * Nrows && -y + Nrows >= 0);

      triplets.push_back(T(y + Nrows, x + Ncols, val));
      triplets.push_back(T(-y + Nrows, -x + Ncols, val));
      count++;
    }
  }

  // // std::cout << "computed: " << count << " entries\n";
  // // debug
  // int cmax = std::numeric_limits<int>::min();
  // int rmax = std::numeric_limits<int>::min();
  // int cmin = std::numeric_limits<int>::max();
  // int rmin = std::numeric_limits<int>::max();
  // for (auto& t : triplets) {
  //   int col = t.col();
  //   int row = t.row();
  //   assert(col < 2*Ncols);
  //   assert(row < 2*Nrows);
  //   cmax = std::max(cmax, col);
  //   rmax = std::max(rmax, row);
  //   cmin = std::min(cmin, col);
  //   rmin = std::min(rmin, row);
  //   //    std::cout << col << " " << row << " " << t.value() << std::endl;
  // }
  // if(triplets.size() > 0) {
  //   std::cout << "x in " << cmin << "\t" << cmax
  //             << std::endl;
  //   std::cout << "y in " << rmin << "\t" << rmax
  //             << std::endl;
  //   std::cout << "nnz: "  << nnz << "\n";
  //   std::cout << "triplets.size: "  << triplets.size() << "\n";
  // } else {
  //   std::cout << "Ooops, triplets were empty!!" << "\n";
  // }

  // assert(triplets.size() <= nnz);
  ft.setFromTriplets(triplets.begin(), triplets.end());
  ft.makeCompressed();
}

// --------------------------------------------------------------------------------
template <template <class> class PSI_RADIAL,
          template <class> class PSI_SPHERICAL,
          template <class> class PSI_SCALING,
          typename TRANSITION_FUNCTION,
          typename NUMERIC>
void
ridge_ft<PSI_RADIAL, PSI_SPHERICAL, PSI_SCALING, TRANSITION_FUNCTION, NUMERIC>::y_ridgelet_ft(
    sparse_mat_t &ft, unsigned int j, int k, unsigned int rho_x, unsigned int rho_y)
{
  x_ridgelet_ft(ft, j, k, rho_y, rho_x);
  ft = ft.transpose();
  ft.makeCompressed();
}

// --------------------------------------------------------------------------------
template <template <class> class PSI_RADIAL,
          template <class> class PSI_SPHERICAL,
          template <class> class PSI_SCALING,
          typename TRANSITION_FUNCTION,
          typename NUMERIC>
void
ridge_ft<PSI_RADIAL, PSI_SPHERICAL, PSI_SCALING, TRANSITION_FUNCTION, NUMERIC>::s_ridgelet_ft(
    mat_t &ft, unsigned int rho_x, unsigned int rho_y)
{
  psi_scaling_t psi_scaling;
  const double dx = 1. / rho_x;
  const double dy = 1. / rho_y;

  ft.resize(4 * rho_y, 4 * rho_x);

  Eigen::RowVectorXd x = Eigen::RowVectorXd::LinSpaced(2 * rho_x - 1, dx, 2 - dx);
  Eigen::RowVectorXd y = Eigen::RowVectorXd::LinSpaced(2 * rho_y - 1, dy, 2 - dy);

  // // debug
  // std::cout << "ridge_ft::s_ridgelet_ft: x= \n" << "\n";
  // std::cout << x << "\n";

  Eigen::RowVectorXd sx(x.size());
  std::transform(x.data(), x.data() + x.size(), sx.data(), psi_scaling);

  Eigen::RowVectorXd sy(y.size());
  std::transform(y.data(), y.data() + y.size(), sy.data(), psi_scaling);

  Eigen::MatrixXd F =
      (y.transpose().replicate(1, x.size()).array() > x.replicate(y.size(), 1).array())
          .select(sy.transpose().replicate(1, x.size()), sx.replicate(y.size(), 1));

  /*
   *
   *     |---+-----------------+-----------+----------|
   *     | 0 | 0               |         0 | 0        |
   *     |---+-----------------+-----------+----------|
   *     | 0 | flipud fliplr F | flip(sy)' | flipud F |
   *     | 0 | flip(sx)        |         1 | sx       |
   *     | 0 | fliplr(F)       |       sy' | F        |
   *     |---+-----------------+-----------+----------|
   *     MATLAB syntax: fliplr -> left-right
   *                    flipud -> up-down
   */
  // std::cout << "F.size : "  << F.rows() << "x" << F.cols() << "\n";
  // std::cout << "ft.rows: "<< ft.rows() << "\n"
  //           << "ft.cols: " << ft.cols() << "\n";

  ft << Eigen::MatrixXd::Zero(1, 4 * rho_x), Eigen::MatrixXd::Zero(y.size(), 1),
      F.rowwise().reverse().colwise().reverse(), sy.reverse().transpose(), F.colwise().reverse(), 0,
      sx.reverse(), 1, sx, Eigen::MatrixXd::Zero(y.size(), 1), F.rowwise().reverse(),
      sy.transpose(), F;
}
