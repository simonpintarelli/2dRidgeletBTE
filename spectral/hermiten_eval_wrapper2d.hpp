#pragma once

#include <tuple>


namespace boltzmann {

// ----------------------------------------------------------------------
/**
 * @brief Evaluator wrapper for 2D normalized physicts' Hermite
 *        polynomials
 *
 */
template <typename POLY_T, typename ELEM>
class HermiteNEvalWrapper2D
{
 public:
  HermiteNEvalWrapper2D(const POLY_T &poly_evald)
      : _poly_evald(poly_evald)
  {
  }

  /**
   * @param e  Element
   * @param q0 Quadrature point ID! in 0-th dimension
   * @param q1 Quadrature point ID! in 1-th dimension
   *
   * @return function element e evaluated at quadrature points
   */
  typename POLY_T::numeric_t operator()(const ELEM &e, unsigned int q0, unsigned int q1) const
  {
    assert((q0 < _poly_evald.get_npoints()) && (q1 < _poly_evald.get_npoints()));
    return _poly_evald.get(get_h0(e).get_degree())[q0] *
           _poly_evald.get(get_h1(e).get_degree())[q1];
  }

 private:
  typedef typename std::tuple_element<0, typename ELEM::container_t>::type h0_t;
  typedef typename std::tuple_element<1, typename ELEM::container_t>::type h1_t;

 private:
  const POLY_T &_poly_evald;

  typename ELEM::Acc::template get<h0_t> get_h0;
  typename ELEM::Acc::template get<h1_t> get_h1;
};

}  // end namespace boltzmann
