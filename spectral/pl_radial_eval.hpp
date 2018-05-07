#pragma once

#include <tuple>

namespace boltzmann {
// ----------------------------------------------------------------------
/**
 * @brief Evaluator wrapper for 2D normalized Laguerre
 *        polynomials
 *
 * @tparam POLY_T LaguerreN, LaguerreNKS
 *
 */
template <typename POLY_T, typename ELEM>
class PLRadialEval
{
 public:
  PLRadialEval(const POLY_T &poly_evald)
      : _poly_evald(poly_evald)
  {
  }

  /**
   * @param e  Element
   * @param q1 Quadrature point !ID! in 1-th dimension
   *
   * @return function element e evaluated at quadrature points
   */
  typename POLY_T::numeric_t operator()(const ELEM &e, unsigned int q) const
  {
    assert(q < _poly_evald.get_npoints());
    const auto &e_rad = get_rad(e);
    return _poly_evald.get(e_rad.get_degree(), e_rad.get_order())[q];
  }

 private:
  // radial element type
  typedef typename std::tuple_element<1, typename ELEM::container_t>::type rad_t;

 private:
  const POLY_T &_poly_evald;
  typename ELEM::Acc::template get<rad_t> get_rad;
};

}  // end namespace boltzmann
