#pragma once

#if (__GNUC__ == 4 && __GNUC_MINOR__ > 8)
#define USE_STD_REGEX
#else
#define USE_BOOST_REGEX
#endif

// system includes ---------------------------------------------------------
#include <fstream>
#include <iostream>

#include <boost/static_assert.hpp>
#include <tuple>

#ifdef USE_STD_REGEX
#include <regex>
#elif defined(USE_BOOST_REGEX)
#include <boost/regex.hpp>
#endif

// own includes ------------------------------------------------------------
#include "enum/enum.hpp"
#include "spectral/basis/spectral_function/hermite_polynomial.hpp"
#include "spectral_basis.hpp"
#include "spectral_basis_factory_base.hpp"
#include "spectral_elem.hpp"
#include "spectral_elem_accessor.hpp"
#include "spectral_function.hpp"


namespace boltzmann {

namespace local_ {
// ----------------------------------------------------------------------
// Hermite basis
class HermiteHX : public HermiteH
{
 public:
  HermiteHX(int k, double w = 0.5)
      : HermiteH(k, w)
  {
  }
  HermiteHX() {}
};

class HermiteHY : public HermiteH
{
 public:
  HermiteHY(int k, double w = 0.5)
      : HermiteH(k, w)
  {
  }
  HermiteHY() {}
};
}  // end namespace local_

// ----------------------------------------------------------------------
class SpectralBasisFactoryHN : public SpectralBasisFactoryBase<local_::HermiteHX, local_::HermiteHY>
{
 public:
  typedef local_::HermiteHX hx_t;
  typedef local_::HermiteHY hy_t;

 public:
  /// definition of element ordering, TODO: define this at a global place
  /**
   *
   *
   * @param basis
   * @param K
   * @param L
   * @param beta
   * @param sorted sort basis functions by `l`, the angular index
   */
  static void create(basis_type &basis, int K, double beta = 2);

  static void write_basis_descriptor(const basis_type &basis,
                                     std::string fname = "spectral_basis.desc");

  /**
   * @brief read basis_descriptor file
   *
   * @param basis
   * @param descriptor_file
   */
  static void create(basis_type &basis, std::string descriptor_file);

  static elem_t make_single_elem(int kx, int ky, double w = 0.5);
};

// ----------------------------------------------------------------------
inline void
SpectralBasisFactoryHN::create(basis_type &basis, int K, double beta)
{
  for (int k = 0; k < K; ++k) {
    for (int s = 0; s <= k; ++s) {
      local_::HermiteHX hx(s, 1 / beta);
      local_::HermiteHY hy(k - s, 1 / beta);
      basis.add_elem(hx, hy);
    }
  }

  basis.finalize();
}

// ----------------------------------------------------------------------
inline void
SpectralBasisFactoryHN::write_basis_descriptor(const basis_type &basis, std::string fname)
{
  // typename elem_t::Acc::template get<fa_type> xir_getter;
  // typename elem_t::Acc::template get<fr_type> rr_getter;

  typename elem_t::Acc::get<fa_type> xir_getter;
  typename elem_t::Acc::get<fr_type> rr_getter;

  std::ofstream fout(fname);
  for (auto it = basis.begin(); it != basis.end(); ++it) {
    fout << xir_getter(*it).get_id() << "\t" << rr_getter(*it).get_id() << std::endl;
  }
  fout.close();
}

// ----------------------------------------------------------------------
inline SpectralBasisFactoryHN::elem_t
SpectralBasisFactoryHN::make_single_elem(int kx, int ky, double w)
{
  local_::HermiteHX hx(kx, w);
  local_::HermiteHY hy(ky, w);
  return elem_t(hx, hy);
}

// ----------------------------------------------------------------------
inline void
SpectralBasisFactoryHN::create(basis_type &basis, std::string descriptor_file)
{
  (void)basis;
  (void)descriptor_file;
  throw std::runtime_error("SpectralBasisFactoryHN::create is not implemented");
}

}  // end namespace boltzmann
