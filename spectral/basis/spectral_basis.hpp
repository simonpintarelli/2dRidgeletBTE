#pragma once

#include <algorithm>
#include <cassert>
#include <map>
#include <unordered_map>
#include <utility>
#include <vector>


namespace boltzmann {

namespace local_ {
// ----------------------------------------------------------------------
template <typename C, typename A, typename... Args>
struct Caller
{
  static void call(C &f, A &&a, Args &&... args)
  {
    f.add_elem(std::forward<A>(a));
    Caller<C, Args...>::call(f, args...);
  }
};

template <typename C, typename A>
struct Caller<C, A>
{
  static void call(C &f, A &&a) { f.add_elem(std::forward<A>(a)); }
};

template <typename C, typename... Args>
void
Call(C &f, Args &&... args)
{
  Caller<C, Args...>::call(f, std::forward<Args>(args)...);
}

}  // end namespace local_

// ----------------------------------------------------------------------
class SpectralBasisDimensionAccessor;

// ----------------------------------------------------------------------
template <typename E>
class SpectralBasis
{
 public:
  typedef E elem_t;

 private:
  typedef std::vector<E> container_t;

 public:
  typedef typename container_t::const_iterator dof_iter_t;
  typedef typename elem_t::id_t id_t;
  typedef unsigned int index_t;
  // typedef std::map<id_t, unsigned int> id_map_t;
  typedef std::unordered_map<id_t, index_t> id_map_t;
  typedef typename elem_t::elem_collection_t elem_collection_t;

 public:
  SpectralBasis(const SpectralBasis &other);
  SpectralBasis(SpectralBasis &&other);
  SpectralBasis()
      : counter(0){};

  index_t get_dof_index(const dof_iter_t &dof_iter) const;
  index_t get_dof_index(const id_t &id) const;
  index_t n_dofs() const { return elements.size(); }
  index_t size() const { return elements.size(); }

  template <typename CMP>
  void sort(const CMP &cmp);

  const elem_t &get_elem(index_t j) const;

  const elem_t &get_elem(const id_t &id) const;

  const dof_iter_t get_iter(const id_t &id) const;

  template <class... Args>
  void add_elem(const Args &... ts);

  dof_iter_t begin() const;

  dof_iter_t end() const;

  // ** access unique elements per type
  const typename elem_collection_t::map_t &get_unique_map() const { return collection.get_map(); }

  /// finalize collection
  void finalize() { collection.finalize(); }

  friend SpectralBasisDimensionAccessor;
  typedef SpectralBasisDimensionAccessor DimAcc;

 private:
  /// stores all elements
  container_t elements;
  /// stores a *unique* collection of the elements in each direction
  elem_collection_t collection;
  /// map index -> element
  id_map_t id_map;
  unsigned int counter;
};

// ----------------------------------------------------------------------
template <typename E>
SpectralBasis<E>::SpectralBasis(const SpectralBasis &other)
    : elements(other.elements)
    , collection(other.collection)
    , id_map(other.id_map)
    , counter(other.counter)
{
  // TODO: improve copy construction and enable move
}

// ----------------------------------------------------------------------
template <typename E>
SpectralBasis<E>::SpectralBasis(SpectralBasis &&other)
    : elements(other.elements)
    , collection(other.collection)
    , id_map(other.id_map)
    , counter(other.counter)
{
  // TODO: improve copy construction and enable move
}

// ----------------------------------------------------------------------
template <typename E>
template <class... Args>
void
SpectralBasis<E>::add_elem(const Args &... ts)
{
  elem_t elem(ts...);
  // collection stores the set of unique elements in each direction
  local_::Call(collection, ts...);
  if (id_map.find(elem.get_id()) != id_map.end()) {
    // this element is already in the basis
  } else {
    this->elements.emplace_back(ts...);
    this->id_map[elem.get_id()] = counter++;
  }
}

// ----------------------------------------------------------------------
template <typename E>
inline const typename SpectralBasis<E>::elem_t &
SpectralBasis<E>::get_elem(index_t j) const
{
  assert(j < elements.size());
  return elements[j];
}

// ----------------------------------------------------------------------
template <typename E>
inline const typename SpectralBasis<E>::elem_t &
SpectralBasis<E>::get_elem(const id_t &id) const
{
  return this->get_elem(this->get_dof_index(id));
}

// ----------------------------------------------------------------------
template <typename E>
inline typename SpectralBasis<E>::dof_iter_t
SpectralBasis<E>::begin() const
{
  return elements.begin();
}

// ----------------------------------------------------------------------
template <typename E>
inline typename SpectralBasis<E>::dof_iter_t
SpectralBasis<E>::end() const
{
  return elements.end();
}

// ----------------------------------------------------------------------
template <typename E>
inline unsigned int
SpectralBasis<E>::get_dof_index(const dof_iter_t &dof_iter) const
{
  return (dof_iter - elements.begin());
}

// ----------------------------------------------------------------------
template <typename E>
inline unsigned int
SpectralBasis<E>::get_dof_index(const id_t &id) const
{
  auto it = id_map.find(id);
  // assert(it != id_map.end());
  if (it == id_map.end()) {
    throw std::runtime_error("SpectralBasis::get_dof_index: elem not found!");
  }
  return it->second;
}

// ----------------------------------------------------------------------
template <typename E>
inline const typename SpectralBasis<E>::dof_iter_t
SpectralBasis<E>::get_iter(const id_t &id) const
{
  auto it = id_map.find(id);
  if (it == id_map.end())
    return elements.end();
  else
    return (elements.begin() + it->second);
}

// ----------------------------------------------------------------------
template <typename E>
template <typename CMP>
inline void
SpectralBasis<E>::sort(const CMP &cmp)
{
  // sort element vector
  std::sort(elements.begin(), elements.end(), cmp);
  // update id_map
  id_map.clear();
  for (unsigned int idx = 0; idx < elements.size(); ++idx) {
    id_map[elements[idx].get_id()] = idx;
  }
}


}  // end namespace boltzmann
