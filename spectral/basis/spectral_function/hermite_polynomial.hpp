#pragma once

#include "spectral/hermiten_impl.hpp"
#include "spectral_function_base.hpp"
#include "spectral_weight_function.hpp"

#include <boost/math/special_functions/gamma.hpp>
#include <stdexcept>


namespace boltzmann {

namespace local_ {
struct hermite_id_t
{
 private:
  constexpr const static double FUZZY = 1e6;

 public:
  typedef hermite_id_t id_t;

  /// Default constructor
  hermite_id_t()
      : hermite_id_t(-1, std::nan("nan"))
  { /* empty */
  }

  hermite_id_t(int k_, double fw_)
      : k(k_)
      , fw(fw_)
      , idw(FUZZY * fw_)
  { /* empty */
  }

  hermite_id_t(const id_t &id)
      : k(id.k)
      , fw(id.fw)
      , idw(id.idw)
  {
  }

  /// degree
  int k;
  /// weight exponent
  double fw;
  /// weight id
  long int idw;

  bool operator<(const id_t &other) const
  {
    return std::tie(k, idw) < std::tie(other.k, other.idw);
  }

  // ----------------------------------------------------------------------
  inline bool operator==(const id_t &other) const
  {
    return std::tie(k, idw) == std::tie(other.k, other.idw);
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
    return "H_" + boost::lexical_cast<std::string>(k) + ", fw_" +
           boost::lexical_cast<std::string>(fw);
  }

  // ----------------------------------------------------------------------
  inline std::tuple<int, long int> key() const { return std::make_tuple(k, idw); }
};
}  // end namespace local_
}  // end namespace boltzmann

namespace std {
// hash functions for id's
template <>
class hash<boltzmann::local_::hermite_id_t>
{
 public:
  size_t operator()(const boltzmann::local_::hermite_id_t &id) const
  {
    std::size_t current = std::hash<double>()(id.fw);
    boost::hash_combine(current, std::hash<int>()(id.k));
    return current;
  }
};
}  // end namespace std

namespace boltzmann {

/**
 * @brief Normalized physicists' Hermite polynomial
 *
 */
class HermiteH : public weighted<HermiteH, true>, public local_::index_policy<local_::hermite_id_t>
{
 public:
  typedef double numeric_t;

 public:
  /**
   * Hermite basis function with exp weight
   *
   * @param k
   * @param w
   *
   * @return
   */
  explicit HermiteH(int k, double w = 0.5);
  explicit HermiteH(const id_t &id)
      : id_(id)
  {
  }
  HermiteH(){};

  /// evaluate polynomial part
  numeric_t evaluate(double x) const;

  /// evaluate weight
  numeric_t weight(double x) const;

  /// return weight
  numeric_t w() const { return id_.fw; }

  const id_t &get_id() const { return id_; }
  unsigned int get_degree() const { return id_.k; }

 private:
  id_t id_;
};

// ----------------------------------------------------------------------
inline HermiteH::HermiteH(int k, double w)
    : id_(k, w)
{ /* empty */
}

// ----------------------------------------------------------------------
inline typename HermiteH::numeric_t
HermiteH::evaluate(double x) const
{
  return boost::math::hermiten(id_.k, x);
}

// ----------------------------------------------------------------------
inline typename HermiteH::numeric_t
HermiteH::weight(double x) const
{
  return std::exp(-x * x * id_.fw);
}

}  // end namespace boltzmann
