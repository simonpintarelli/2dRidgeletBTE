#pragma once

#include "spectral/basis/spectral_elem.hpp"
#include "traits/type_traits.hpp"


namespace boltzmann {
template <typename AngularBasisFunction, typename RadialBasisFunction>
struct SpectralBasisFactoryBase
{
  typedef AngularBasisFunction fa_type;
  typedef RadialBasisFunction fr_type;

  typedef
      typename numeric_super_type<typename fa_type::numeric_t, typename fr_type::numeric_t>::type
          numeric_t;
  typedef SpectralElem<numeric_t, fa_type, fr_type> elem_t;

  typedef SpectralBasis<elem_t> basis_type;
};
}
