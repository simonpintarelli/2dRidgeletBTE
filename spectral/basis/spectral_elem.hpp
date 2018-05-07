#pragma once

#include <boost/fusion/container/generation/make_map.hpp>
#include <boost/fusion/include/make_map.hpp>
#include <boost/mpl/vector.hpp>
#include <tuple>

#include <boost/fusion/algorithm/auxiliary/copy.hpp>
#include <boost/fusion/include/copy.hpp>
// own includes ---------------------------------------------------------
#include "spectral/basis/spectral_elem_accessor.hpp"
#include "spectral/basis/spectral_elem_collection.hpp"
#include "spectral/basis/spectral_function/spectral_id.hpp"


namespace boltzmann {

/// forward declaration
class SpectralElemAccessor;

// ----------------------------------------------------------------------
template <typename RTYPE, class... Args>
class SpectralElem
{
 public:
  typedef RTYPE numeric_t;
  typedef std::tuple<Args...> container_t;
  typedef boost::mpl::vector<Args...> types_t;
  typedef spectral_id<Args...> id_t;

  // define a typedef for later usage
  typedef SpectralElemCollection<Args...> elem_collection_t;

 public:
  SpectralElem(const Args &... elems);

  SpectralElem(const SpectralElem &other);
  /// copy assignment operator
  const SpectralElem &operator=(const SpectralElem &other);

  template <typename C, typename... Cs>
  numeric_t evaluate(const C &c, const Cs &... cs) const;
  constexpr int evaluate() const { return 1; }

  template <typename C, typename... Cs>
  numeric_t evaluate_weighted(const C &c, const Cs &... cs) const;

  const id_t &get_id() const { return id_; }
  const id_t &id() const { return id_; }

  constexpr int evaluate_weighted() const { return 1; }

 private:
  // tuple of
  container_t elements;
  /// boost fusion map type
  id_t id_;

  friend SpectralElemAccessor;

 public:
  typedef SpectralElemAccessor Acc;
};

// ----------------------------------------------------------------------
template <typename RTYPE, class... Args>
SpectralElem<RTYPE, Args...>::SpectralElem(const SpectralElem &other)
    : elements(other.elements)
{
  boost::fusion::copy(other.id_, id_);
}

// ----------------------------------------------------------------------
template <typename RTYPE, class... Args>
const SpectralElem<RTYPE, Args...> &
SpectralElem<RTYPE, Args...>::operator=(const SpectralElem &other)
{
  elements = other.elements;
  boost::fusion::copy(other.id_, id_);
  return *this;
}

// ----------------------------------------------------------------------
template <typename RTYPE, class... Args>
SpectralElem<RTYPE, Args...>::SpectralElem(const Args &... elems)
    : elements(elems...)
{
  auto tmp = boost::fusion::make_map<Args...>(elems.get_id()...);
  // without fusion::copy id_ ends up in an undefined state
  boost::fusion::copy(tmp, id_);
}

// ----------------------------------------------------------------------
template <typename RTYPE, class... Args>
template <typename C, typename... Cs>
inline typename SpectralElem<RTYPE, Args...>::numeric_t
SpectralElem<RTYPE, Args...>::evaluate(const C &c, const Cs &... cs) const
{
  constexpr unsigned int items = sizeof...(Cs);
  constexpr unsigned int dim = sizeof...(Args);
  constexpr unsigned int i = dim - (items + 1);

  return std::get<i>(this->elements).evaluate(c) * this->evaluate(cs...);
}

// ----------------------------------------------------------------------
template <typename RTYPE, class... Args>
template <typename C, typename... Cs>
inline typename SpectralElem<RTYPE, Args...>::numeric_t
SpectralElem<RTYPE, Args...>::evaluate_weighted(const C &c, const Cs &... cs) const
{
  constexpr unsigned int items = sizeof...(Cs);
  constexpr unsigned int dim = sizeof...(Args);
  constexpr unsigned int i = dim - (items + 1);

  return std::get<i>(this->elements).evaluate(c) * std::get<i>(this->elements).weight(c) *
         this->evaluate_weighted(cs...);
}
}  // end namespace boltzmann
