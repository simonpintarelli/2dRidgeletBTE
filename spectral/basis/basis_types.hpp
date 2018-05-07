#pragma once

#include "spectral_basis_factory_hermite.hpp"
#include "spectral_basis_factory_ks.hpp"

namespace boltzmann {

typedef SpectralBasisFactoryHN::basis_type hermite_basis_t;
/// type representing Hermite h_i(x)
typedef SpectralBasisFactoryHN::hx_t HermiteHX_t;
/// type representing Hermite h_j(y)
typedef SpectralBasisFactoryHN::hy_t HermiteHY_t;

typedef SpectralBasisFactoryKS::basis_type pl_basis_t;

}  // boltzmann
