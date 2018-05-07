#pragma once

#include <boost/fusion/adapted/std_tuple.hpp>
#include <boost/fusion/include/begin.hpp>
#include <boost/fusion/include/filter_view.hpp>
#include <boost/fusion/sequence/intrinsic/begin.hpp>
#include <boost/fusion/view/filter_view.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/mpl/same_as.hpp>
#include <boost/static_assert.hpp>

#include <tuple>


namespace boltzmann {

struct SpectralElemAccessor
{
  template <typename E>
  struct get
  {
    template <typename T>
    const E &operator()(const T &t) const
    {
      typedef decltype(t.elements) C_t;
      typedef boost::fusion::filter_view<C_t const, boost::mpl::same_as<E>> filter_t;
      filter_t filter(t.elements);
      // // TODO: insert a static assert that filter is of length one
      // // BOOST_STATIC_ASSERT(boost::fusion::size<filter_t>::value == 1);
      return boost::fusion::deref(boost::fusion::begin(filter));
    }
  };
};
}
