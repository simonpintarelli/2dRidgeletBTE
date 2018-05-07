#pragma once

#include <boost/fusion/container/map.hpp>
#include <boost/fusion/container/map/map_fwd.hpp>
#include <boost/fusion/include/at_key.hpp>
#include <boost/fusion/include/map.hpp>
#include <boost/fusion/include/map_fwd.hpp>
#include <boost/fusion/include/zip_view.hpp>
#include <boost/fusion/sequence/intrinsic/at_key.hpp>
#include <boost/fusion/view/zip_view.hpp>
#include <boost/mpl/for_each.hpp>


namespace boltzmann {
class SpectralBasisDimensionAccessor
{
 public:
  /**
   * @brief Get a map<index, Element> along dimension E
   *
   * @param b SpectralBasis
   * @tparam E ...
   * @return
   */
  template <typename E>
  struct get_id_map
  {
    template <typename B>
    auto operator()(const B &b) const -> decltype(boost::fusion::at_key<E>(b.collection.get_map()))
    {
      return boost::fusion::at_key<E>(b.collection.get_map());
    }
  };

  template <typename E>
  struct get_vec
  {
    template <typename B>
    auto operator()(const B &b) const -> decltype(boost::fusion::at_key<E>(b.collection.get_vec()))
    {
      return boost::fusion::at_key<E>(b.collection.get_vec());
    }
  };
};
}  // end namepsace boltzmann
