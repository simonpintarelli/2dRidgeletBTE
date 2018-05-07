#pragma once

#include <boost/functional/hash.hpp>
#include <boost/lexical_cast.hpp>
#include <complex>
#include <functional>
#include <iostream>
#include <tuple>
// own includes ----------------------------------------------------------------
#include "enum/enum.hpp"


namespace boltzmann {
namespace local_ {
template <typename T>
struct index_policy
{
  typedef T id_t;
};
}

}  // end namespace std
