#pragma once

#include <boost/assert.hpp>
#include "lambda.hpp"


unsigned int
translation_size(enum rt_type l, int j, int rho = 1, int dim = 0)
{
  BOOST_VERIFY(dim == 0 || dim == 1);

  unsigned int tx;

  if (dim == 0) {
    switch (l) {
      case rt_type::S:
        tx = 4 * rho;
        break;

      case rt_type::X:
        tx = std::pow(2, j + 2) * rho;
        break;

      case rt_type::Y:
        tx = 8 * rho;
        break;

      case rt_type::D:
        tx = std::pow(2, j + 2) * rho;
        break;
      default:
        tx = 0;
        break;
    }
  } else /*dim == 1*/ {
    switch (l) {
      case rt_type::S: {
        tx = 4 * rho;
        break;
      }
      case rt_type::X: {
        tx = 8 * rho;
        break;
      }
      case rt_type::Y: {
        tx = std::pow(2, j + 2) * rho;
        break;
      }
      case rt_type::D: {
        tx = 8 * rho;
        break;
      }
      default:
        break;
    }
  }

  return tx;
}
