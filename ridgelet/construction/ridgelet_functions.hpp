#pragma once

#include <array>
#include <base/polyval.hpp>
#include <boost/math/constants/constants.hpp>


class TransitionFunction
{
 public:
  template <typename NUMERIC>
  NUMERIC operator()(const NUMERIC &x) const;
};

template <typename NUMERIC>
inline NUMERIC
TransitionFunction::operator()(const NUMERIC &x) const
{
  constexpr static std::array<int, 8> coeffs_ = {0, 0, 0, 0, 35, -84, 70, -20};

  if (x >= 1)
    return 1;
  else if (x <= 0)
    return 0;
  else {
    double v = polyval(coeffs_.begin(), x, coeffs_.size());
    assert(v >= 0 && v <= 1);
    return v;
  }
}

// --------------------------------------------------------------------------------
template <typename TF = TransitionFunction>
class PsiRadial1
{
 public:
  template <typename NUMERIC>
  NUMERIC operator()(const NUMERIC &x) const;

 private:
  TF t_;
};

template <typename TF>
template <typename NUMERIC>
inline NUMERIC
PsiRadial1<TF>::operator()(const NUMERIC &x) const
{
  using namespace boost::math::constants;

  if (std::abs(x) >= 1 && std::abs(x) <= 2) {
    return std::sin(pi<NUMERIC>() / 2 * t_(std::abs(x) - 1));
  } else if (2 < std::abs(x) && std::abs(x) < 4) {
    return std::cos(pi<NUMERIC>() / 2 * t_(0.5 * std::abs(x) - 1));
  } else {
    return 0;
  }
}

// --------------------------------------------------------------------------------
template <typename TF = TransitionFunction>
class PsiSpherical1
{
 public:
  template <typename NUMERIC>
  NUMERIC operator()(const NUMERIC &x) const;

 private:
  TF t_;
};

template <typename TF>
template <typename NUMERIC>
inline NUMERIC
PsiSpherical1<TF>::operator()(const NUMERIC &x) const
{
  if (x <= 0)
    return std::sqrt(t_(1 + x));
  else
    return std::sqrt(t_(1 - x));
}

// --------------------------------------------------------------------------------
/**
 * @brief \f$ \Psi_{0,s,0} \f$
 *
 */
template <typename TF = TransitionFunction>
class PsiScaling1
{
 public:
  template <typename NUMERIC>
  NUMERIC operator()(const NUMERIC &x, const NUMERIC &y) const;

  template <typename NUMERIC>
  NUMERIC operator()(const NUMERIC &z) const;

 private:
  TF t_;
};

// --------------------------------------------------------------------------------
template <typename TF>
template <typename NUMERIC>
inline NUMERIC
PsiScaling1<TF>::operator()(const NUMERIC &x, const NUMERIC &y) const
{
  using namespace boost::math::constants;
  NUMERIC z = std::max(std::abs(x), std::abs(y));
  if (z < 1)
    return 1;
  else if (z >= 1 && z <= 2)
    return std::cos(pi<NUMERIC>() / 2 * t(z - 1));
  else
    return 0;
}

// --------------------------------------------------------------------------------
template <typename TF>
template <typename NUMERIC>
inline NUMERIC
PsiScaling1<TF>::operator()(const NUMERIC &z) const
{
  using namespace boost::math::constants;
  if (z < 1)
    return 1;
  else if (z >= 1 && z <= 2)
    return std::cos(pi<NUMERIC>() / 2 * t_(z - 1));
  else
    return 0;
}
