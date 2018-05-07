#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <boost/math/constants/constants.hpp>


/*
 *
 *  Remember that:
 *   o----> V_x
 *   |       A              B
 *   |       o--------------o------o A''
 *   v       |              |      |
 *    V_y    |              |      |
 *           |              |      |
 *           |              |      |
 *           |              |      |
 *           |              |      |
 *         D o--------------o      o D'
 *           |              C      |
 *           |                     |
 *           +--------------o------o A'
 *                          B'
 *
 */
template <typename MATRIX>
void
make_inflow_source(MATRIX &dst, double vx, double vy, double Lx, double Ly, double ql, double qt)
{
  constexpr const double PI = boost::math::constants::pi<double>();
  int Nx = dst.rows();
  int Ny = dst.cols();

  if (!(Lx > 1) || !(Ly > 1)) throw std::runtime_error("no absorbing layer existent");

  auto xi = Eigen::ArrayXd::LinSpaced(Nx, 0, Lx).transpose();
  auto yi = Eigen::ArrayXd::LinSpaced(Ny, 0, Ly);
  Eigen::ArrayXXd X = xi.replicate(Ny, 1);
  Eigen::ArrayXXd Y = yi.replicate(1, Nx);
  Eigen::Vector2d v = {vy, vx};
  v = v / v.norm();
  Eigen::Vector2d c = {1, 1};  // corner of active domain

  assert(X.rows() == Ny && X.cols() == Nx);
  assert(Y.rows() == Ny && Y.cols() == Nx);

  Eigen::Vector2d Dp = {1.0, Lx};  // D'
  Eigen::Vector2d Ap = {Ly, Lx};   // A'
  Eigen::Vector2d Bp = {Ly, 1.0};  // B'
  typedef Eigen::Hyperplane<double, 2> hyperplane_t;
  Eigen::Rotation2D<double> rot90(PI / 2);

  Eigen::Vector2d n = rot90 * v;

  dst.setZero();

  if (vx > 0 && vy > 0) {
    /* inflow from left and top boundary  */
    hyperplane_t lD(n, Dp);
    hyperplane_t lA(n, Ap);
    hyperplane_t lB(n, Bp);

    for (int i = 0; i < Ny; ++i) {
      for (int j = 0; j < Nx; ++j) {
        Eigen::Vector2d x = {Y(i, j), X(i, j)};
        if (lD.signedDistance(x) > 0) {
          dst(i, j) = ql;
        } else if (lA.signedDistance(x) > 0) {
          dst(i, j) = qt;
        } else if (lB.signedDistance(x) > 0) {
          dst(i, j) = ql;
        } else {
          dst(i, j) = qt;
        };
      }
    }
  } else if (vx > 0) {
    /*  inflow only on left boundary  */
    hyperplane_t hyperplane(n, c);
    for (int i = 0; i < Ny; ++i) {
      for (int j = 0; j < Nx; ++j) {
        Eigen::Vector2d x = {Y(i, j), X(i, j)};
        double d = hyperplane.signedDistance(x);
        if (d > 0 || X(i, j) >= 1) dst(i, j) = ql;
      }
    }
  } else if (vy > 0) {
    /*  inflow only on top boundary  */
    hyperplane_t lC(n, c);
    for (int i = 0; i < Ny; ++i) {
      for (int j = 0; j < Nx; ++j) {
        Eigen::Vector2d x = {Y(i, j), X(i, j)};
        if (Y(i, j) > 1 || lC.signedDistance(x) < 0) dst(i, j) = qt;
      }
    }
  } else {
    // Quadrant 4 => no inflow
  }
}
