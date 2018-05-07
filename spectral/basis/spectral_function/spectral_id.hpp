#pragma once

#include <boost/fusion/container/map.hpp>
#include <boost/fusion/container/map/map_fwd.hpp>
#include <boost/fusion/include/at_key.hpp>
#include <boost/fusion/include/map.hpp>
#include <boost/fusion/include/map_fwd.hpp>
#include <boost/fusion/mpl.hpp>
#include <boost/fusion/sequence/intrinsic/at_key.hpp>
//#include <map>
#include <boost/mpl/for_each.hpp>
#include <iostream>
#include <tuple>
#include <unordered_map>

#include <boost/fusion/include/zip_view.hpp>
#include <boost/fusion/view/zip_view.hpp>

namespace boltzmann {
/**
 * @brief spectral index type
 *        implements less than and equal for boost::map
 *
 * @param other
 *
 * @return
 */ template <typename... Args>
class spectral_id : public boost::fusion::map<boost::fusion::pair<Args, typename Args::id_t>...>
{
 private:
  /// weak ordering helpers
  //@{
  struct equal_
  {
    template <class>
    struct result;

    template <class F, class T, class U>
    struct result<F(T, U)>
    {
      typedef bool type;
    };

    template <typename T>
    bool operator()(bool t, const T &tup)
    {
      return boost::fusion::at_c<0>(tup) == boost::fusion::at_c<1>(tup) && t;
    }
  };

  struct less_than_helper_
  {
    template <class>
    struct result;

    template <class F, class T, class U>
    struct result<F(T, U)>
    {
      typedef bool type;
    };

    template <typename T>
    bool operator()(bool t, const T &tup)
    {
      return (boost::fusion::at_c<0>(tup) < boost::fusion::at_c<1>(tup)) ||
             ((boost::fusion::at_c<0>(tup) == boost::fusion::at_c<1>(tup)) && t);
    }
  };
  //@}

  struct toString_
  {
    template <class>
    struct result;
    template <class F, class T, class U>
    struct result<F(T, U)>
    {
      typedef std::string type;
    };

    template <typename T>
    std::string operator()(std::string ss, const T &t)
    {
      return ss + " " + t.second.to_string();
    }
  };

 public:
  typedef boost::fusion::map<boost::fusion::pair<Args, typename Args::id_t>...> base_type;

  bool operator==(const base_type &other) const
  {
    typedef boost::fusion::vector<const base_type &, const base_type &> sequences;
    auto zipped = boost::fusion::zip_view<sequences>(sequences(*this, other));
    return boost::fusion::fold(zipped, true, equal_());
  }

  /**
   * this implementation is inefficient
   *
   * @param other
   *
   * @return
   */
  bool operator<(const base_type &other) const
  {
    typedef boost::fusion::vector<const base_type &, const base_type &> sequences;
    auto zipped = boost::fusion::zip_view<sequences>(sequences(*this, other));
    return (boost::fusion::fold(zipped, false, less_than_helper_()) && !((*this) == other));
  }

  std::string to_string() const { return boost::fusion::fold(*this, "", toString_()); }

};
}  // end namespace boltzmann

namespace std {
template <typename... ARGS>
class hash<boltzmann::spectral_id<ARGS...>>
{
  struct hasher_
  {
    template <class>
    struct result;
    template <class F, class T, class U>
    struct result<F(T, U)>
    {
      typedef size_t type;
    };
    template <typename T>
    size_t operator()(size_t h, const T &t)
    {
      typedef typename T::second_type value_type;
      return std::hash<value_type>()(t.second) ^ h;
    }
  };

 private:
  typedef boltzmann::spectral_id<ARGS...> type;

 public:
  size_t operator()(const type &value) const { return boost::fusion::fold(value, 0, hasher_()); }
};

}  // end std
