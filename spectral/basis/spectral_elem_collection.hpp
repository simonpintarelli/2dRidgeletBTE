#pragma once

#include <boost/fusion/algorithm/iteration/for_each.hpp>
#include <boost/fusion/container/map.hpp>
#include <boost/fusion/container/map/map_fwd.hpp>
#include <boost/fusion/include/at_key.hpp>
#include <boost/fusion/include/for_each.hpp>
#include <boost/fusion/include/map.hpp>
#include <boost/fusion/include/map_fwd.hpp>
#include <boost/fusion/include/zip_view.hpp>
#include <boost/fusion/sequence/intrinsic/at_key.hpp>
#include <boost/fusion/view/zip_view.hpp>
#include <boost/mpl/for_each.hpp>
#include <cassert>

#include <map>
#include "spectral/basis/spectral_function.hpp"

namespace boltzmann {

/**
 * Stores subelements (i.e. functions in phi, r coord. as std::map or std::vec)
 */
template <class... Args>
class SpectralElemCollection
{
 public:
  typedef boost::fusion::map<boost::fusion::pair<Args, std::map<typename Args::id_t, Args>>...>
      map_t;
  typedef boost::fusion::map<boost::fusion::pair<Args, std::vector<Args>>...> vec_t;

  const map_t &get_map() const
  {
    assert(is_finalized_ == true);
    return elems_;
  }
  const vec_t &get_vec() const
  {
    assert(is_finalized_ == true);
    return vec_;
  }

 private:
  typedef SpectralElemCollection<Args...> this_type;

 public:
  SpectralElemCollection()
      : finalizer_(this)
      , is_finalized_(false)
  {
  }

  // ----------------------------------------------------------------------
  // copy constructor
  SpectralElemCollection(const this_type &other)
      : finalizer_(other.finalizer_)
      , is_finalized_(other.is_finalized_)
  {
    boost::fusion::copy(other.elems_, elems_);
    boost::fusion::copy(other.vec_, vec_);
  }

  // ----------------------------------------------------------------------
  template <typename ELEM>
  void add_elem(const ELEM &elem)
  {
    assert(!is_finalized_);
    auto &elem_map = boost::fusion::at_key<ELEM>(elems_);
    if (elem_map.find(elem.get_id()) != elem_map.end()) {
      // element is already stored => do nothing
    } else {
      elem_map[elem.get_id()] = elem;
    }
  }

  // ----------------------------------------------------------------------
  void finalize()
  {
    assert(is_finalized_ == false);
    boost::fusion::for_each(elems_, finalizer_);
    is_finalized_ = true;
  }

 protected:
  struct finalizer_t
  {
    finalizer_t(this_type *ref)
        : ref_(ref)
    {
    }
    template <typename T>
    void operator()(const T &t) const
    {
      typedef typename T::first_type index_type;
      for (auto it = t.second.begin(); it != t.second.end(); ++it) {
        boost::fusion::at_key<index_type>(ref_->vec_).push_back(it->second);
      }
    }
    this_type *ref_;
  };

  finalizer_t finalizer_;
  map_t elems_;
  vec_t vec_;
  bool is_finalized_;
};
}
