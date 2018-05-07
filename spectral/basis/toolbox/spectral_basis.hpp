#pragma once

// system includes ------------------------------------------------------------
#include <Eigen/Sparse>
#include <boost/iterator/filter_iterator.hpp>
#include <boost/mpl/identity.hpp>
#include <functional>
#include <iterator>
#include <tuple>

// own includes ------------------------------------------------------------
#include "spectral/basis/spectral_basis.hpp"
#include "spectral/basis/spectral_elem_accessor.hpp"


namespace spectral {
// ----------------------------------------------------------------------
template <typename BASIS>
unsigned int
get_max_l(const BASIS &basis)
{
  typedef typename std::tuple_element<0, typename BASIS::elem_t::container_t>::type elem_t;
  // radial basis
  typename BASIS::elem_t::Acc::template get<elem_t> get_xir;
  unsigned int maxL = 0;
  for (auto it = basis.begin(); it != basis.end(); ++it) {
    unsigned int l = get_xir(*it).get_id().l;
    if (l > maxL) maxL = l;
  }

  return maxL;
}

// ----------------------------------------------------------------------
/**
 * @Brief Return the max polynomial degree for the polar basis
 *
 * TODO, works only for the polar basis
 *
 * @param basis
 *
 * @return
 */
template <typename BASIS>
unsigned int
get_max_k(const BASIS &basis)
{
  typedef typename std::tuple_element<1, typename BASIS::elem_t::container_t>::type elem_t;
  typename BASIS::elem_t::Acc::template get<elem_t> get_phi;
  unsigned int maxK = 0;
  for (auto it = basis.begin(); it != basis.end(); ++it) {
    unsigned int k = get_phi(*it).get_id().k;
    if (k > maxK) maxK = k;
  }
  return maxK;
}

/**
 *  @brief max. polynomial degree + 1
 *
 *  @param spectral basis
 *  @return return type
 */
template <typename BASIS>
unsigned int
get_K(const BASIS &basis)
{
  return get_max_k(basis) + 1;
}

}  // end namespace spectral_basis
