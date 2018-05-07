#pragma once

#include "qgauss_laguerre.hpp"
#include "qmaxwell.hpp"
#include "qmidpoint.hpp"

#include "tensor_product_quadrature.hpp"


namespace boltzmann {

template <template <class, class> class TensorQuadT, typename QSpherical, typename QRadial>
class QuadFactoryBase
{
 public:
  struct descriptor
  {
    descriptor(int na_, int nr_, double alpha_)
        : na(na_)
        , nr(nr_)
        , alpha(alpha_)
    {
    }
    int na;
    int nr;
    double alpha;

    bool operator<(const descriptor& other) const
    {
      return std::tie(na, nr, alpha) < std::tie(other.na, other.nr, other.alpha);
    }

    bool operator==(const descriptor& other) const
    {
      return std::tie(na, nr, alpha) == std::tie(other.na, other.nr, other.alpha);
    }
  };

  typedef TensorQuadT<QSpherical, QRadial> type;

  static type* create(const descriptor& desc)
  {
    QSpherical qm(desc.na);
    QRadial qr(desc.alpha, desc.nr);

    return (new type(qm, qr));
  }
};

template <class>
class QuadFactory
{
};

// ----------------------------------------------------------------------
template <>
class QuadFactory<TensorProductQuadrature<QMidpoint, QGaussLaguerre> >
    : public QuadFactoryBase<TensorProductQuadrature, QMidpoint, QGaussLaguerre>
{
};

// ----------------------------------------------------------------------
template <>
class QuadFactory<TensorProductQuadratureC<QMidpoint, QGaussLaguerre> >
    : public QuadFactoryBase<TensorProductQuadratureC, QMidpoint, QGaussLaguerre>
{
};

// ----------------------------------------------------------------------
template <>
class QuadFactory<TensorProductQuadrature<QMidpoint, QMaxwell> >
    : public QuadFactoryBase<TensorProductQuadrature, QMidpoint, QMaxwell>
{
};

// ----------------------------------------------------------------------
template <>
class QuadFactory<TensorProductQuadratureC<QMidpoint, QMaxwell> >
    : public QuadFactoryBase<TensorProductQuadratureC, QMidpoint, QMaxwell>
{
};

}  // end boltzmann
