#pragma once

#include <tuple>
#include "ridgelet/lambda.hpp"
#include "ridgelet/ridgelet_frame.hpp"


inline std::tuple<int, int>
tgrid_dim(const lambda_t &lam, const RidgeletFrame &rf)
{
  int Tx = 0;
  int Ty = 0;
  auto j = lam.j;
  auto t = lam.t;
  int rx = rf.rho_x();
  int ry = rf.rho_y();

  switch (t) {
    case rt_type::S: {
      Tx = 4 * rx;
      break;
    }
    case rt_type::X: {
      Tx = std::pow(2, j + 2) * rx;
      break;
    }
    case rt_type::Y: {
      Tx = 8 * rx;
      break;
    }
    case rt_type::D: {
      Tx = std::pow(2, j + 2) * rx;
      break;
    }
    default:
      assert(false);
      break;  // should throw an error here
  }

  switch (t) {
    case rt_type::S: {
      Ty = 4 * ry;
      break;
    }
    case rt_type::X: {
      Ty = 8 * ry;
      break;
    }
    case rt_type::Y: {
      Ty = std::pow(2, j + 2) * ry;
      break;
    }
    case rt_type::D: {
      Ty = 8 * ry;
      break;
    }
    default:
      assert(false);
      break;  // should throw an error here
  }

  return std::make_tuple(Tx, Ty);
}
