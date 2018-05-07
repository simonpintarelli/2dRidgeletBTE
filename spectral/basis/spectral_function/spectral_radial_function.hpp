#pragma once

// system includes ---------------------------------------------------------
#include <boost/math/special_functions/laguerre.hpp>
#include <functional>

// own includes ------------------------------------------------------------
#include "spectral_function_base.hpp"
#include "spectral_weight_function.hpp"


namespace boltzmann {

namespace local_ {
struct laguerre_id_t
{
 private:
  constexpr const static double FUZZY = 1e6;

 public:
  typedef laguerre_id_t id_t;

  /// Default constructor
  laguerre_id_t()
      : fw(-1)
      , k(-1)
      , idw(-1)
  {
  }
  laguerre_id_t(double fw_, int k_)
      : fw(fw_)
      , k(k_)
      , idw(FUZZY * fw_)
  {
  }
  // explicit laguerre_id_t(const id_t& id) : fw(id.fw), k(id.k), idw(id.idw) {}

  /// weight exponent
  double fw;
  /// Laguerre polynomial index
  int k;

  bool operator<(const laguerre_id_t &other) const
  {
    return std::tie(k, idw) < std::tie(other.k, other.idw);
    //#warning "fix this!"
    //    return k < other.k;
  }

  bool operator==(const laguerre_id_t &other) const
  {
    return std::tie(k, idw) == std::tie(other.k, other.idw);
    //#warning "fix this!"
    //    return k == other.k;
  }

  friend std::ostream &operator<<(std::ostream &stream, const laguerre_id_t &x)
  {
    stream << x.to_string();
    return stream;
  }

  std::string to_string() const
  {
    return "(fw_" + boost::lexical_cast<std::string>(fw) + ", k_" +
           boost::lexical_cast<std::string>(k) + ") ";
  }

  std::tuple<int, long int> key() const { return std::make_tuple(k, idw); }

  /// weight id
  long int idw;
};
}  // end namespace local_
}  // end namespace boltzmann

namespace std {
// hash functions for id's
template <>
class hash<boltzmann::local_::laguerre_id_t>
{
 public:
  size_t operator()(const boltzmann::local_::laguerre_id_t &id) const
  {
    std::size_t current = std::hash<double>()(id.fw);
    boost::hash_combine(current, std::hash<int>()(id.k));
    return current;
  }
};
}  // end namespace std

namespace boltzmann {
// --------------------------------------------------------------------------------
class LaguerreRR : public weighted<LaguerreRR, true>,
                   public local_::index_policy<local_::laguerre_id_t>
{
 public:
  typedef double numeric_t;

 public:
  LaguerreRR(double fw_, int k_)
      : id_(fw_, k_)
  {
  }

  explicit LaguerreRR(const id_t &id)
      : id_(id)
  {
  }

  LaguerreRR()
      : id_(-1, -1)
  {
  }

  /// set the exponential weight
  // void set_exponent(double f);

  /// evaluate polynomial part
  double evaluate(double r) const;

  /// evaluate weight
  double weight(double r) const;

  /// return weight
  double w() const { return id_.fw; }

  const id_t &get_id() const { return id_; }
  // const double get_beta() const __attribute__ ((deprecated))
  // { return id_.beta; }

 private:
  double evenk(int k, double r) const;
  double oddk(int k, double r) const;

  id_t id_;
};

// ----------------------------------------------------------------------
inline double
LaguerreRR::evaluate(double r) const
{
  if (id_.k % 2 == 0)
    return this->evenk(id_.k / 2, r);
  else
    return this->oddk((id_.k - 1) / 2, r);
}

// ----------------------------------------------------------------------
inline double
LaguerreRR::weight(double r) const
{
  return std::exp(-r * r * id_.fw);
}

// ----------------------------------------------------------------------
inline double
LaguerreRR::evenk(int kk, double r) const
{
  return boost::math::laguerre(kk, 0, r * r);
}

// ----------------------------------------------------------------------
inline double
LaguerreRR::oddk(int kk, double r) const
{
  return std::sqrt(1. / (kk + 1.)) * r * boost::math::laguerre(kk, 1, r * r);
}
}  // end namespace boltzmann
// ----------------------------------------------------------------------
