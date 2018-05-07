#pragma once

#include <boost/mpl/if.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include <boost/type_traits.hpp>
#include <complex>


namespace boltzmann {

/*
 *  @brief returns T1 if it is a complex type, otherwise returns type of T2
 *
 */
template <typename T1, typename T2>
struct numeric_super_type
{
  typedef typename boost::mpl::eval_if_c<boost::is_complex<T1>::value, /* is a complex type */
                                         boost::mpl::identity<T1>,
                                         boost::mpl::if_c<boost::is_complex<T2>::value,
                                                          T2, /* is a complex type */
                                                          T2  /* a numeric type */
                                                          > >::type type;
};

// ----------------------------------------------------------------------
// MULTIPRECISION, boost::mpfr

template <typename T>
struct is_mpfr
{
  static const bool value = false;
};

/**
 *   specialization for boost mpfr floating type
 */
template <unsigned N, boost::multiprecision::expression_template_option ET>
struct is_mpfr<boost::multiprecision::number<boost::multiprecision::mpfr_float_backend<N>, ET> >
{
  static const bool value = true;
};

/**
 *   specialization for boost mpfr floating type: const reference
 */
template <unsigned N, boost::multiprecision::expression_template_option ET>
struct is_mpfr<
    const boost::multiprecision::number<boost::multiprecision::mpfr_float_backend<N>, ET>&>
{
  static const bool value = true;
};

/**
 *   specialization for boost mpfr floating type: reference
 */
template <unsigned N, boost::multiprecision::expression_template_option ET>
struct is_mpfr<boost::multiprecision::number<boost::multiprecision::mpfr_float_backend<N>, ET>&>
{
  static const bool value = true;
};

}  // end namespace boltzmann
