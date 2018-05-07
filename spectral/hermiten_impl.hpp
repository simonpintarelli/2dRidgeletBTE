#pragma once


#include <boost/math/constants/constants.hpp>
#include <boost/math/policies/error_handling.hpp>
#include <boost/math/special_functions/math_fwd.hpp>
#include <boost/math/tools/config.hpp>
#include <cmath>

#include "spectral/mpfr/import_std_math.hpp"

namespace boost {
namespace math {

// Recurrance relation for Hermite polynomials:
template <class T1, class T2, class T3>
inline typename tools::promote_args<T1, T2, T3>::type
hermiten_next(unsigned n, T1 x, T2 Hn, T3 Hnm1)
{
  typedef T1 numeric_t;

  const numeric_t fn = 2 / numeric_t(n + 1);
  const numeric_t fnm = numeric_t(n) / (n + 1);
  return ::math::sqrt(fn) * x * Hn - ::math::sqrt(fnm) * Hnm1;
}

namespace detail {

// Implement Hermite polynomials via recurrance:
template <class T>
T
hermiten_imp(unsigned n, T x)
{
  static const T pi = boost::math::constants::pi<T>();
  static const T pif = ::math::pow(pi, (T)-0.25);
  T p0 = pif;

  if (n == 0) return p0;

  T p1 = sqrt(T(2)) * x * pif;

  unsigned c = 1;

  while (c < n) {
    std::swap(p0, p1);
    p1 = hermiten_next(c, x, p0, p1);
    ++c;
  }
  return p1;
}

}  // namespace detail

template <class T, class Policy>
inline typename tools::promote_args<T>::type
hermiten(unsigned n, T x, const Policy &)
{
  typedef typename tools::promote_args<T>::type result_type;
  typedef typename policies::evaluation<result_type, Policy>::type value_type;
  return policies::checked_narrowing_cast<result_type, Policy>(
      detail::hermiten_imp(n, static_cast<value_type>(x)),
      "boost::math::hermiten<%1%>(unsigned, %1%)");
}

template <class T>
inline typename tools::promote_args<T>::type
hermiten(unsigned n, T x)
{
  return boost::math::hermiten(n, x, policies::policy<>());
}

}  // namespace math
}  // namespace boost
