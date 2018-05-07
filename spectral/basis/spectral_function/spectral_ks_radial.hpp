#pragma once

#include <boost/functional/hash.hpp>
#include <boost/math/special_functions/binomial.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/laguerre.hpp>
#include <stdexcept>

#include "spectral/basis/spectral_function/spectral_function_base.hpp"
#include "spectral/basis/spectral_function/spectral_weight_function.hpp"


namespace boltzmann {

namespace local_ {
struct laguerreKS_id_t
{
 private:
  constexpr const static double FUZZY = 1e6;

 public:
  typedef laguerreKS_id_t id_t;

  /// Default constructor
  laguerreKS_id_t()
      : j(std::nan("nan"))
      , k(std::nan("nan"))
      , fw(std::nan("nan"))
      , idw(std::nan("nan"))
  {
  }

  laguerreKS_id_t(int k_, int j_, double fw_)
      : j(j_)
      , k(k_)
      , fw(fw_)
      , idw(FUZZY * fw_)
  {
  }

  laguerreKS_id_t(const id_t &id)
      : j(id.j)
      , k(id.k)
      , fw(id.fw)
      , idw(id.idw)
  {
  }

  /// order parameter
  int j;
  /// Polynomial degree
  int k;
  /// weight exponent
  double fw;
  /// weight id
  long int idw;

  bool operator<(const id_t &other) const
  {
    return std::tie(j, k, idw) < std::tie(other.j, other.k, other.idw);
    //#warning "fix this!"
    //    return k < other.k;
  }

  // ----------------------------------------------------------------------
  inline bool operator==(const id_t &other) const
  {
    return std::tie(j, k, idw) == std::tie(other.j, other.k, other.idw);
    //#warning "fix this!"
    //    return k == other.k;
  }

  // ----------------------------------------------------------------------
  friend std::ostream &operator<<(std::ostream &stream, const id_t &x)
  {
    stream << x.to_string();
    return stream;
  }

  // ----------------------------------------------------------------------
  std::string to_string() const
  {
    return "(fw_" + boost::lexical_cast<std::string>(fw) + ", k_" +
           boost::lexical_cast<std::string>(k) + ", j_" + boost::lexical_cast<std::string>(j) +
           ") ";
  }

  // ----------------------------------------------------------------------
  inline std::tuple<int, int, long int> key() const { return std::make_tuple(j, k, idw); }
};
}  // end namespace local_
}  // end namespace boltzmann

namespace std {
// hash functions for id's
template <>
class hash<boltzmann::local_::laguerreKS_id_t>
{
 public:
  size_t operator()(const boltzmann::local_::laguerreKS_id_t &id) const
  {
    std::size_t current = std::hash<double>()(id.fw);
    boost::hash_combine(current, std::hash<int>()(id.k));
    boost::hash_combine(current, std::hash<int>()(id.j));
    return current;
  }
};
}  // end namespace std

namespace boltzmann {

/**
 * @brief Laguerre Radial Polynomial used by Kitzler & Schoeberl
 *
 * Normalization & Orthogonality
 * =============================
 *
 * For \f$ k_1 \equiv k_2 \operatorname{mod} 2 \f$, \f$k_1\f$ even:
 * \f[
 *   \int_0^{\infty} r^{2j}  r^{2j} L^{(2j)}_{\frac{k_1}{2}-j}(r^2)
 * L^{(2j)}_{\frac{k_2}{2}-j} (r^2) e^{-r^2}\; r \operatorname{d} r= \frac{1}{2}
 * \delta_{k_1, k_2}
 * \f]
 *
 *  For \f$ k_1 \equiv k_2 \operatorname{mod} 2 \f$, \f$k_1\f$ odd:
 * \f[
 *   \int_0^{\infty} r^{2j+1}  r^{2j+1} L^{(2j+1)}_{\frac{k_1-1}{2}-j}(r^2)
 * L^{(2j+1)}_{\frac{k_2-1}{2}-j} (r^2) e^{-r^2}\; r \operatorname{d} r=
 * \frac{1}{2} \delta_{k_1,
 * k_2}
 * \f]
 *
 */
class LaguerreKS : public weighted<LaguerreKS, true>,
                   public local_::index_policy<local_::laguerreKS_id_t>
{
 public:
  typedef double numeric_t;

 public:
  /**
   * @brief Basis functions of the form \f$ r^{2*j + k\mod2}
   * L_{(k-k\mod2)/2-j}^{2*j + k\mod2}(r^2)
   * \f$
   *
   *
   * @param k  polynomial degree
   * @param j  angular parameter
   * @param w  weight
   *
   * @return
   */
  explicit LaguerreKS(int k, int j, double w = 0.5);
  explicit LaguerreKS(const id_t &id)
      : id_(id)
  {
  }
  LaguerreKS(){};

  /// evaluate polynomial part
  numeric_t evaluate(double r) const;

  /// evaluate weight
  numeric_t weight(double r) const;

  /// return weight
  numeric_t w() const { return id_.fw; }

  const id_t &get_id() const { return id_; }

  unsigned int get_order() const { return 2 * id_.j + id_.k % 2; }
  unsigned int get_degree() const { return id_.k / 2 - id_.j; }

 private:
  id_t id_;
};

// ----------------------------------------------------------------------
inline LaguerreKS::LaguerreKS(int k, int j, double w)
    : id_(k, j, w)
{
  if (j > k / 2) {
    std::runtime_error("invalid combination of parameters in LaguerreKS");
  }
}

// ----------------------------------------------------------------------
inline typename LaguerreKS::numeric_t
LaguerreKS::evaluate(double r) const
{
  // normalization factor
  int alpha = 2 * id_.j + (id_.k % 2);
  int n = id_.k / 2 - id_.j;

  double a = 1;
  for (int i = n + alpha; i > n; --i) {
    a *= i;
  }

  return std::pow(r, 2 * id_.j + (id_.k % 2)) * boost::math::laguerre(n, alpha, r * r) /
         std::sqrt(a);
}

// ----------------------------------------------------------------------
inline typename LaguerreKS::numeric_t
LaguerreKS::weight(double r) const
{
  return std::exp(-r * r * id_.fw);
}

}  // end namespace boltzmann
