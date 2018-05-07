
//  (C) Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_SPECIAL_LAGUERREN_HPP
#define BOOST_MATH_SPECIAL_LAGUERREN_HPP

#include "spectral/mpfr/import_std_math.hpp"

#include <boost/math/policies/error_handling.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/laguerre.hpp>
#include <boost/math/special_functions/math_fwd.hpp>
#include <boost/math/tools/config.hpp>

namespace boost {
namespace math {

// forward declarations
template <class T, class Policy>
typename tools::promote_args<T>::type
laguerren(unsigned n, unsigned m, T x, const Policy &pol);

template <class T1, class T2>
typename laguerre_result<T1, T2>::type
laguerren(unsigned n, T1 m, T2 x);

// // Recurrance relation for Laguerre polynomials:
// template <class T1, class T2, class T3>
// inline typename tools::promote_args<T1, T2, T3>::type
// laguerre_next(unsigned n, T1 x, T2 Ln, T3 Lnm1)
// {
//   typedef typename tools::promote_args<T1, T2, T3>::type result_type;
//   return ((2 * n + 1 - result_type(x)) * result_type(Ln) - n *
//   result_type(Lnm1)) / (n + 1);
// }

namespace detail {

// // Implement Laguerre polynomials via recurrance:
// template <class T>
// T laguerre_imp(unsigned n, T x)
// {
//   T p0 = 1;
//   T p1 = 1 - x;

//   if(n == 0)
//     return p0;

//   unsigned c = 1;

//   while(c < n)
//     {
//       std::swap(p0, p1);
//       p1 = laguerre_next(c, x, p0, p1);
//       ++c;
//     }
//   return p1;
// }

// template <class T, class Policy>
// inline typename tools::promote_args<T>::type
// laguerre(unsigned n, T x, const Policy&, const mpl::true_&)
// {
//   typedef typename tools::promote_args<T>::type result_type;
//   typedef typename policies::evaluation<result_type, Policy>::type
//   value_type;
//   return policies::checked_narrowing_cast<result_type,
//   Policy>(detail::laguerre_imp(n,
//   static_cast<value_type>(x)), "boost::math::laguerre<%1%>(unsigned, %1%)");
// }

template <class T>
inline typename tools::promote_args<T>::type
laguerren(unsigned n, unsigned m, T x, const mpl::false_ &)
{
  return boost::math::laguerren(n, m, x, policies::policy<>());
}

}  // namespace detail

// template <class T>
// inline typename tools::promote_args<T>::type
// laguerre(unsigned n, T x)
// {
//   return laguerre(n, x, policies::policy<>());
// }

// Recurrence for associated polynomials:
template <class T1, class T2, class T3>
inline typename tools::promote_args<T1, T2, T3>::type
laguerren_next(unsigned n, unsigned l, T1 x, T2 Pl, T3 Plm1)
{
  typedef typename tools::promote_args<T1, T2, T3>::type result_type;

  // return ((2*n+1 + l -x) / result_type(n+1) * result_type(Pl) -
  //         (n+l)/ result_type(n+1) *  result_type(Plm1) );

  result_type nn = n;
  result_type mm = l;
  return (Pl * (mm + 2 * nn - x + 1) / ::math::sqrt((nn + 1) * (mm + nn + 1)) -
          Plm1 * nn * (nn + mm) / ::math::sqrt(nn * (nn + 1) * (nn + mm) * (mm + nn + 1)));
}

namespace detail {
// Laguerre Associated Polynomial:
template <class T, class Policy>
T
laguerren_imp(unsigned n, unsigned m, T x, const Policy &pol)
{
  // Special cases:
  if (m == 0) return boost::math::laguerre(n, x, pol);

  // normalization
  T a = 1;
  for (unsigned int i = m + n; i > n; --i) {
    a *= i;
  }

  T p0 = 1 / ::math::sqrt(a);

  if (n == 0) return p0;

  T p1 = (m + 1 - x) / ::math::sqrt(a);

  unsigned c = 1;

  while (c < n) {
    std::swap(p0, p1);
    p1 = laguerren_next(c, m, x, p0, p1);
    ++c;
  }
  return p1;
}
}

template <class T, class Policy>
inline typename tools::promote_args<T>::type
laguerren(unsigned n, unsigned m, T x, const Policy &pol)
{
  typedef typename tools::promote_args<T>::type result_type;
  typedef typename policies::evaluation<result_type, Policy>::type value_type;
  // return policies::checked_narrowing_cast<result_type,
  // Policy>(detail::laguerren_imp(n, m,
  // static_cast<value_type>(x), pol),
  //                                                              "boost::math::laguerren<%1%>(unsigned,
  //                                                              unsigned,
  //                                                              %1%)");

  return detail::laguerren_imp(n, m, static_cast<value_type>(x), pol);
}

template <class T1, class T2>
inline typename laguerre_result<T1, T2>::type
laguerren(unsigned n, T1 m, T2 x)
{
  typedef typename policies::is_policy<T2>::type tag_type;
  return detail::laguerren(n, m, x, tag_type());
}

}  // namespace math
}  // namespace boost

#endif  // BOOST_MATH_SPECIAL_LAGUERREN_HPP
