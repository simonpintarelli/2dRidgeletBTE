#pragma once

#include "spectral_coord_traits.hpp"


namespace boltzmann {

// --------------------------------------------------------------------------------
template <typename F, bool T>
struct weighted
{
};

// --------------------------------------------------------------------------------
template <typename F>
struct weighted<F, true>
{
  typename SpectralCoordTraits<F>::return_type weight(
      const typename SpectralCoordTraits<F>::coord_type &c) const
  {
    return static_cast<const F &>(*this).weight(c);
  }
};

// --------------------------------------------------------------------------------
template <typename F>
struct weighted<F, false>
{
  constexpr typename SpectralCoordTraits<F>::return_type weight(
      const typename SpectralCoordTraits<F>::coord_type &c) const
  {
    return 1;
  }
};

}  // end namespace boltzmann
