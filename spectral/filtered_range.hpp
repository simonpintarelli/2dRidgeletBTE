#pragma once

#include <boost/iterator/filter_iterator.hpp>
#include <boost/mpl/identity.hpp>

#include <functional>
#include <tuple>


namespace boltzmann {

template <typename ITERATOR, typename ELEM>
std::tuple<boost::filter_iterator<std::function<bool(const ELEM &)>, ITERATOR>,
           boost::filter_iterator<std::function<bool(const ELEM &)>, ITERATOR>>
filtered_range(const ITERATOR &begin,
               const typename boost::mpl::identity<ITERATOR>::type &end,
               const std::function<bool(const ELEM &)> &f)
{

  return std::make_tuple(boost::make_filter_iterator(f, begin, end),
                         boost::make_filter_iterator(f, end, end));
}

}  // end namespace boltzmann
