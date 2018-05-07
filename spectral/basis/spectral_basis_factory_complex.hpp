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

#ifdef USE_STD_REGEX
#include <regex>
#elif defined(USE_BOOST_REGEX)
#include <boost/regex.hpp>
#endif

// own includes ------------------------------------------------------------
#include "enum/enum.hpp"
#include "spectral_basis.hpp"
#include "spectral_basis_factory_base.hpp"
#include "spectral_elem.hpp"
#include "spectral_elem_accessor.hpp"
#include "spectral_function.hpp"


namespace boltzmann {

namespace local_ {
template <typename BASIS>
struct CMP
{
  /**
   * @brief lexicographical ordering for basis elements
   *
   *
   * @return
   */
  template <typename E>
  bool operator()(const E &e1, const E &e2) const
  {
    auto id1 = e1.get_id();
    auto id2 = e2.get_id();

    if (id1 < id2)
      return true;
    else
      return false;
  }
};
}  // end namespace local_

// ----------------------------------------------------------------------
class SpectralBasisFactoryComplex : public SpectralBasisFactoryBase<XiRC, LaguerreRR>
{
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
  static void create(basis_type &basis, int K, int L, double beta, bool sorted = false);
  /**
   *
   *
   * @param basis
   * @param K
   * @param L
   * @param beta
   * @param sorted sort basis functions by `l`, the angular index
   */
  static void create_test(basis_type &basis, int K, int L, double beta, bool sorted = false);

  static void write_basis_descriptor(const basis_type &basis,
                                     std::string fname = "spectral_basis.desc");

  /**
   * @brief read basis_descriptor file
   *
   * @param basis
   * @param descriptor_file
   */
  static void create(basis_type &basis, std::string descriptor_file);
};

// ----------------------------------------------------------------------
void
SpectralBasisFactoryComplex::create(basis_type &basis, int K, int L, double beta, bool sorted)
{
  for (int l = -L; l < L; ++l) {
    for (int k = 0; k < K; ++k) {
      if (k % 2 == std::abs(l) % 2) {
        fa_type xir(l);
        fr_type phi(1. / beta, k);

        basis.add_elem(xir, phi);
      }
    }
  }

  /// sort basis functions by l-index
  // if (sorted) {
  //   basis.sort(local_::CMP<basis_type>());
  // }

  basis.finalize();
}

// ----------------------------------------------------------------------
void
SpectralBasisFactoryComplex::create_test(basis_type &basis, int K, int L, double beta, bool sorted)
{
  for (int l = -L; l < L; ++l) {
    for (int k = 0; k < K; ++k) {
      if (k % 2 == std::abs(l) % 2) {
        fa_type xir(l);
        double fw = 1 / beta;
        /// momentum conservation
        if (k == 1 && l == 1) fw = 0;
        /// mass and energy conservation
        if ((k == 0 || k == 2) && l == 0) fw = 0;
        fr_type phi(fw, k);
        basis.add_elem(xir, phi);
      }
    }
  }

  // /// sort basis functions by l-index
  // if(sorted) {
  //   basis.sort(local_::CMP<basis_type>());
  // }
  basis.finalize();
}

// ----------------------------------------------------------------------
void
SpectralBasisFactoryComplex::write_basis_descriptor(const basis_type &basis, std::string fname)
{
  typename elem_t::Acc::template get<fa_type> xir_getter;
  typename elem_t::Acc::template get<fr_type> rr_getter;

  std::ofstream fout(fname);
  for (auto it = basis.begin(); it != basis.end(); ++it) {
    fout << xir_getter(*it).get_id() << "\t" << rr_getter(*it).get_id() << std::endl;
  }
  fout.close();
}

// ----------------------------------------------------------------------
void
SpectralBasisFactoryComplex::create(basis_type &basis, std::string descriptor_file)
{
  std::ifstream ifile;
  ifile.open(descriptor_file);
  std::string line;

#ifdef USE_STD_REGEX
  std::regex basisf_regex(
      "exp_ii_([-]?[0-9]+)[[:space:]]*[(]beta_([0-9.]+),[[:"
      "space:]]*k_([0-9]+)[)]");
  // new descriptor
  std::regex basisf_regex_new(
      "exp_ii_([-]?[0-9.]+)[[:space:]]*[(]fw_([0-9.]+),"
      "[[:space:]]*k_([0-9]+)[)]");
  std::smatch basisf_match;

  while (std::getline(ifile, line)) {
    std::regex_search(line, basisf_match, basisf_regex);
    if (basisf_match.size() > 0) {
      auto l_match = basisf_match[1];
      auto beta_match = basisf_match[2];
      auto k_match = basisf_match[3];
      int l = atoi(l_match.str().c_str());
      int k = atoi(k_match.str().c_str());
      double beta = atof(beta_match.str().c_str());

      fr_type phi(1. / beta, k);

      fa_type xir(l);
      basis.add_elem(xir, phi);
    }
    // try to match new descriptor
    std::regex_search(line, basisf_match, basisf_regex_new);
    if (basisf_match.size() > 0) {
      // angular frequency
      auto l_match = basisf_match[1];
      int l = atoi(l_match.str().c_str());
      // exponential weight
      auto fw_match = basisf_match[2];
      double fw = atof(fw_match.str().c_str());
      // laguerre poly index
      auto k_match = basisf_match[3];
      int k = atoi(k_match.str().c_str());

      fr_type phi(fw, k);
      fa_type xir(l);
      basis.add_elem(xir, phi);
    }
  }
#elif defined(USE_BOOST_REGEX)
  boost::regex basisf_regex_new(
      "exp_ii_([-]?[0-9]+)[[:space:]]*[(]fw_([0-9.]+)"
      ",[[:space:]]*k_([0-9]+)[)]",
      boost::regex_constants::ECMAScript);
  boost::regex basisf_regex(
      "exp_ii_([-]?[0-9]+)[[:space:]]*[(]beta_([0-9.]+),["
      "[:space:]]*k_([0-9]+)[)]",
      boost::regex_constants::ECMAScript);
  boost::cmatch match;
  bool is_matched;
  while (std::getline(ifile, line)) {
    is_matched = boost::regex_search(line.c_str(), match, basisf_regex);
    if (is_matched) {
      auto l_match = match[1];
      auto beta_match = match[2];
      auto k_match = match[3];
      int l = atoi(l_match.str().c_str());
      int k = atoi(k_match.str().c_str());
      double beta = atof(beta_match.str().c_str());

      fr_type phi(1. / beta, k);
      fa_type xir(l);
      basis.add_elem(xir, phi);
    }
    // try to match new descriptor
    is_matched = boost::regex_search(line.c_str(), match, basisf_regex_new);
    if (is_matched) {
      // angular frequency
      auto l_match = match[1];
      int l = atoi(l_match.str().c_str());
      // exponential weight
      auto fw_match = match[2];
      double fw = atof(fw_match.str().c_str());
      // laguerre poly index
      auto k_match = match[3];
      int k = atoi(k_match.str().c_str());

      fr_type phi(fw, k);
      fa_type xir(l);
      basis.add_elem(xir, phi);
    }
  }
#else
  // cannot parse basis descriptor
  BOOST_STATIC_ASSERT(false);
#endif
  basis.finalize();
  ifile.close();
}

}  // end namespace boltzmann
