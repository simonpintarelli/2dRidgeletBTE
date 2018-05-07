#pragma once

#include <boost/functional/hash.hpp>
#include <functional>
#include <tuple>


namespace std {

template <typename... TTypes>
class hash<std::tuple<TTypes...>>
{
 private:
  typedef std::tuple<TTypes...> Tuple;

  template <int N>
  size_t operator()(const Tuple &value __attribute__((unused))) const
  {
    return 0;
  }

  template <int N, typename THead, typename... TTail>
  size_t operator()(const Tuple &value) const
  {
    constexpr int Index = N - sizeof...(TTail) - 1;
    std::size_t current = std::hash<THead>()(std::get<Index>(value));
    boost::hash_combine(current, operator()<N, TTail...>(value));

    return current;
  }

 public:
  size_t operator()(const Tuple &value) const
  {
    return operator()<sizeof...(TTypes), TTypes...>(value);
  }
};
}
