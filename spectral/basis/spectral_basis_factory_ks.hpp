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
#include "base/exceptions.hpp"

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
struct CMPKS
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
    int l1 = get_xi(e1).get_id().l;
    int l2 = get_xi(e2).get_id().l;

    int t1 = get_xi(e1).get_id().t;
    int t2 = get_xi(e2).get_id().t;

    int k1 = get_phi(e1).get_id().k;
    int k2 = get_phi(e2).get_id().k;

    int j1 = get_phi(e1).get_id().j;
    int j2 = get_phi(e2).get_id().j;

    return std::tie(l1, t1, k1, j1) < std::tie(l2, t2, k2, j2);
  }

 private:
  // angle elment type
  typedef typename std::tuple_element<0, typename BASIS::elem_t::container_t>::type ea_t;
  // radial element type
  typedef typename std::tuple_element<1, typename BASIS::elem_t::container_t>::type er_t;

  typename BASIS::elem_t::Acc::template get<ea_t> get_xi;
  typename BASIS::elem_t::Acc::template get<er_t> get_phi;
};
}  // end namespace local_

// ----------------------------------------------------------------------
class SpectralBasisFactoryKS : public SpectralBasisFactoryBase<XiR, LaguerreKS>
{
 public:
  /**
 *  @brief create the Polar-Laguerre basis with polynomial degree <K, sort basis
 * by (l, t, k)
 *         l: angular index, t: sin/cos, k: polynomial degree
 *
 *  Detailed description
 *
 *  @param[out] basis
 *  @param[in]  K
 */
  static void create(basis_type &basis, int K);
  /// definition of element ordering, TODO: define this at a global place
  /**
   *
   *
   * @param[out] basis
   * @param[in] K
   * @param[in] L
   * @param[in] beta
   * @param[in] sorted sort basis functions by `l`, the angular index
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

  static elem_t make_elem(int j, int k, enum TRIG t, double w = 0.5);
  static elem_t get_mass_elem();
  static elem_t get_ux_elem();
  static elem_t get_uy_elem();
  static elem_t get_energy_elem();

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
inline typename SpectralBasisFactoryKS::elem_t
SpectralBasisFactoryKS::make_elem(int j, int k, enum TRIG t, double w)
{
  // if ( j > k/2  || (t == TRIG::SIN && j == 0))
  //   throw std::runtime_error("wrong arguments");

  fa_type fa(t, 2 * j + k % 2);
  fr_type fr(k, j, w);
  return elem_t(fa, fr);
}

inline typename SpectralBasisFactoryKS::elem_t
SpectralBasisFactoryKS::get_mass_elem()
{
  return SpectralBasisFactoryKS::make_elem(0, 0, TRIG::COS, 0);
}

inline typename SpectralBasisFactoryKS::elem_t
SpectralBasisFactoryKS::get_ux_elem()
{
  return SpectralBasisFactoryKS::make_elem(0, 1, TRIG::COS, 0);
}

inline typename SpectralBasisFactoryKS::elem_t
SpectralBasisFactoryKS::get_uy_elem()
{
  return SpectralBasisFactoryKS::make_elem(0, 1, TRIG::SIN, 0);
}

inline typename SpectralBasisFactoryKS::elem_t
SpectralBasisFactoryKS::get_energy_elem()
{
  return SpectralBasisFactoryKS::make_elem(0, 2, TRIG::COS, 0);
}

// ----------------------------------------------------------------------
inline void
SpectralBasisFactoryKS::create(basis_type &basis, int K)
{
  double beta = 2;
  bool sorted = true;
  SpectralBasisFactoryKS::create(basis, K, K, beta, sorted);
}

// ----------------------------------------------------------------------
inline void
SpectralBasisFactoryKS::create(basis_type &basis, int K, int L, double beta, bool sorted)
{
  for (int k = 0; k < K; ++k) {
    // sin
    for (int j = 1 - k % 2; j <= std::min(k / 2, L); ++j) {
      fa_type fa(TRIG::SIN, 2 * j + k % 2);
      fr_type fr(k, j, 1 / beta);
      basis.add_elem(fa, fr);
    }

    // cos
    for (int j = 0; j <= std::min(k / 2, L); ++j) {
      fa_type fa(TRIG::COS, 2 * j + k % 2);
      fr_type fr(k, j, 1 / beta);
      basis.add_elem(fa, fr);
    }
  }

  /// sort basis functions by l-index
  if (sorted) {
    basis.sort(local_::CMPKS<basis_type>());
  }

  basis.finalize();
}

// ----------------------------------------------------------------------
/**
 *  @brief Create test basis for Petrov-Galerkin Ansatz
 *
 *  @param K create basis of polynomial degree < K
 *  @param
 *  @return return type
 */
inline void
SpectralBasisFactoryKS::create_test(basis_type &basis, int K, int L, double beta, bool sorted)
{
  for (int k = 0; k < K; ++k) {
    // sin
    for (int j = 1 - k % 2; j <= std::min(k / 2, L); ++j) {
      fa_type fa(TRIG::SIN, 2 * j + k % 2);

      double w = 1 / beta;
      if (j == 0 && k == 1) w = 0.0;
      fr_type fr(k, j, w);
      basis.add_elem(fa, fr);
    }

    // cos
    for (int j = 0; j <= std::min(k / 2, L); ++j) {
      fa_type fa(TRIG::COS, 2 * j + k % 2);

      double w = 1 / beta;
      if (j == 0 && k == 1) w = 0.0;
      if (j == 0 && (k == 0 || k == 2)) w = 0.0;
      fr_type fr(k, j, w);
      basis.add_elem(fa, fr);
    }
  }

  /// sort basis functions by l-index
  if (sorted) {
    basis.sort(local_::CMPKS<basis_type>());
  }

  basis.finalize();
}

// ----------------------------------------------------------------------
inline void
SpectralBasisFactoryKS::write_basis_descriptor(const basis_type &basis, std::string fname)
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
inline void
SpectralBasisFactoryKS::create(basis_type &basis, std::string descriptor_file)
{
  std::ifstream ifile;
  ifile.open(descriptor_file);
  ASSERT(ifile.is_open());

  std::string line;

#ifdef USE_STD_REGEX
  // new descriptor
  std::regex basisf_regex_new(
      "(cos|sin)_([0-9.]+)[[:space:]]*[(]fw_([0-9.]+),["
      "[:space:]]*k_([0-9]+),[[:space:]]*j_([0-9]+)"
      "[)]");
  std::smatch basisf_match;

  while (std::getline(ifile, line)) {
    // try to match new descriptor
    std::regex_search(line, basisf_match, basisf_regex_new);
    if (basisf_match.size() > 0) {
      auto sincos_match = basisf_match[1];
      // angular frequency
      auto l_match = basisf_match[2];
      int l = atoi(l_match.str().c_str());
      // exponential weight
      auto fw_match = basisf_match[3];
      double fw = atof(fw_match.str().c_str());
      // laguerre poly index
      auto k_match = basisf_match[4];
      int k = atoi(k_match.str().c_str());
      // coupled angular index
      auto j_match = basisf_match[5];
      int j = atoi(j_match.str().c_str());

      fr_type phi(k, j, fw);
      if (sincos_match.str().compare("sin") == 0) {
        fa_type xir(TRIG::SIN, l);
        basis.add_elem(xir, phi);
      } else {
        fa_type xir(TRIG::COS, l);
        basis.add_elem(xir, phi);
      }
    }
  }

#elif defined(USE_BOOST_REGEX)
  boost::regex basisf_regex_new(
      "(cos|sin)_([0-9.]+)[[:space:]]*[(]fw_([0-9.]+)"
      ",[[:space:]]*k_([0-9]+),[[:space:]]*j_([0-9]+)"
      "[)]",
      boost::regex_constants::ECMAScript);

  boost::cmatch match;
  bool is_matched;
  while (std::getline(ifile, line)) {
    // try to match new descriptor
    is_matched = boost::regex_search(line.c_str(), match, basisf_regex_new);
    if (is_matched) {
      auto sincos_match = match[1];
      // angular frequency
      auto l_match = match[2];
      int l = atoi(l_match.str().c_str());
      // exponential weight
      auto fw_match = match[3];
      double fw = atof(fw_match.str().c_str());
      // laguerre poly index
      auto k_match = match[4];
      int k = atoi(k_match.str().c_str());

      // coupled angular index
      auto j_match = match[5];
      int j = atoi(j_match.str().c_str());

      fr_type phi(k, j, fw);
      if (sincos_match.str().compare("sin") == 0) {
        fa_type xir(TRIG::SIN, l);
        basis.add_elem(xir, phi);
      } else {
        fa_type xir(TRIG::COS, l);
        basis.add_elem(xir, phi);
      }
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
