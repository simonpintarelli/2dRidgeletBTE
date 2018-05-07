#pragma once

#include <base/types.hpp>
#include <boost/functional/hash.hpp>
#include <functional>
#include <tuple>


struct lambda_t
{
  lambda_t() {}

  explicit lambda_t(unsigned int j_, rt_type t_, int k_)
      : j(j_)
      , t(t_)
      , k(k_)
  { /* empty */
  }

  explicit lambda_t(unsigned int j_, char t_, int k_)
  {
    rt_type _t;
    switch (t_) {
      case 's': {
        _t = rt_type::S;
        break;
      }
      case 'x': {
        _t = rt_type::X;
        break;
      }
      case 'y': {
        _t = rt_type::Y;
        break;
      }
      case 'd': {
        _t = rt_type::D;
        break;
      }
      default:
        throw 0;
        break;
    }
    j = j_;
    t = _t;
    k = k_;
  }

  int j;
  enum rt_type t;
  int k;

  bool operator==(const lambda_t &other) const
  {
    return (other.j == j && other.t == t && other.k == k);
  }

  friend std::ostream &operator<<(std::ostream &s, const lambda_t &lambda)
  {
    s << "(j=" << lambda.j << ",";
    switch (lambda.t) {
      case rt_type::X: {
        s << "X";
        break;
      }
      case rt_type::Y: {
        s << "Y";
        break;
      }
      case rt_type::D: {
        s << "D";
        break;
      }
      case rt_type::S: {
        s << "S";
        break;
      }
      default:
        s << "UNKOWN";
        break;
    }
    s << ",k=" << lambda.k << ")";

    return s;
  }

  explicit operator std::string() const
  {
    std::stringstream ss;
    ss << *this;
    return ss.str();
  }

  std::string toString() const { return (std::string)(*this); }

  bool operator<(const lambda_t &other) const
  {
    return std::tie(j, t, k) < std::tie(other.j, other.t, other.k);
  }
};

// --------------------------------------------------------------------------------
// implement std::hash for lambda_t;
namespace std {

template <>
class hash<lambda_t>
{
 private:
  typedef lambda_t value_t;

 public:
  std::size_t operator()(const value_t &value) const
  {
    std::size_t current = std::hash<unsigned int>{}(value.j);
    boost::hash_combine(current, std::hash<int>{}((int)value.t));
    boost::hash_combine(current, std::hash<int>{}(value.k));
    return current;
  }
};
}
