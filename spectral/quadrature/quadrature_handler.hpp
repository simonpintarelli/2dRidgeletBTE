#pragma once

#include <complex>
#include <memory>
#include <map>
#include <vector>

#include "quad_factory.hpp"

namespace boltzmann {

/**
 * @brief Storage class for R^2 quadrature rules with different exponential weighting
 *
 * @param key
 *
 * @return
 */
template <typename QUAD = TensorProductQuadrature<QMidpoint, QMaxwell> >
class QuadratureHandler
{
 public:
  typedef QUAD quad_t;

 private:
  typedef QuadFactory<quad_t> factory_t;

 public:
  typedef typename factory_t::descriptor descriptor;

 private:
  typedef std::shared_ptr<quad_t> quad_ptr_t;
  typedef std::map<descriptor, quad_ptr_t> quad_map_t;

 public:
  const quad_t& get_quad(const descriptor& key);

  /**
   *
   * @param alpha   exponential weight
   * @param na      #quad. points in angle
   * @param nr      #quad. points in radius
   *
   * @return quadrature for R^2
   */
  const quad_t& get_quad(double alpha, int na, int nr);

 private:
  quad_map_t quad_map_;
};

// ------------------------------------------------------------
template <typename QUAD>
const typename QuadratureHandler<QUAD>::quad_t&
QuadratureHandler<QUAD>::get_quad(const descriptor& key)
{
  auto it = quad_map_.find(key);
  if (it != quad_map_.end())
    return *(it->second);
  else {
    // construct this element
    typedef QuadFactory<quad_t> factory_t;
    quad_ptr_t quad(factory_t::create(key));
    quad_map_[key] = quad;
    return *quad;
  }
}

// ------------------------------------------------------------
template <typename QUAD>
const typename QuadratureHandler<QUAD>::quad_t&
QuadratureHandler<QUAD>::get_quad(double alpha, int na, int nr)
{
  const descriptor key(na, nr, alpha);
  return this->get_quad(key);
}

}  // end boltzmann
