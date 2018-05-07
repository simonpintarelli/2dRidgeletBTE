#pragma once

#include "spectral_function_base.hpp"

namespace boltzmann {
// indexing class
namespace local_ {
struct xir_id_t
{
  typedef xir_id_t id_t;
  xir_id_t()
      : t(-1)
      , l(-1)
  {
  }
  explicit xir_id_t(int t_, int l_)
      : t(t_)
      , l(l_)
  {
  }
  xir_id_t(const id_t &id)
      : t(id.t)
      , l(id.l)
  {
  }

  int t;
  int l;

  inline bool operator<(const xir_id_t &other) const
  {
    return std::tie(t, l) < std::tie(other.t, other.l);
  }

  inline bool operator==(const xir_id_t &other) const { return (t == other.t) && (l == other.l); }

  friend std::ostream &operator<<(std::ostream &stream, const xir_id_t &x)
  {
    stream << x.to_string();  //(x.t == 0 ? "cos_" : "sin_") << x.l << " ";
    return stream;
  }

  std::string to_string() const
  {
    return (t == TRIG::COS ? "cos_" : "sin_") + boost::lexical_cast<std::string>(l);
  }
};
}  // end namespace local_
}  // end namespace boltzmann

namespace std {
template <>
class hash<boltzmann::local_::xir_id_t>
{
 public:
  inline size_t operator()(const boltzmann::local_::xir_id_t &id) const
  {
    std::size_t current = std::hash<int>()(id.t);
    boost::hash_combine(current, std::hash<int>()(id.l));
    return current;
  }
};
}  // end namespace std

namespace boltzmann {
// --------------------------------------------------------------------------------
class XiR : public weighted<XiR, false>, public local_::index_policy<local_::xir_id_t>
{
 public:
  typedef double numeric_t;

 public:
  // TODO make t of class type enum {SIN, COS}
  explicit XiR(int t_, int l_)
      : id_(t_, l_)
  {
  }
  XiR()
      : id_(-1, -1)
  {
  }
  explicit XiR(const id_t &id)
      : id_(id)
  {
  }

  inline double evaluate(double phi) const
  {
    if (id_.t == TRIG::COS)
      return std::cos(double(id_.l) * phi);
    else if (id_.t == TRIG::SIN)
      return std::sin(double(id_.l) * phi);
    else {
      assert(false);
      return std::nan("1");
    }
  }

  const id_t &get_id() const { return id_; }

 private:
  id_t id_;
};

}  // end namespace boltzmann
