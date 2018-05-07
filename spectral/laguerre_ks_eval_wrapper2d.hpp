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
class LaguerreNEvalWrapper2D
{
 public:
  LaguerreNEvalWrapper2D(const POLY_T &poly_evald)
      : _poly_evald(poly_evald)
  {
  }

  /**
   * @param e  Element
   * @param dummy alias `q0` Quadrature point !ID! in 0-th dimension
   * @param q1 Quadrature point !ID! in 1-th dimension
   *
   * @return function element e evaluated at quadrature points
   */
  typename POLY_T::numeric_t operator()(const ELEM &e,
                                        unsigned int dummy __attribute__((unused)),
                                        unsigned int q1) const
  {
    assert(q1 < _poly_evald.get_npoints());
    const auto &e_rad = get_rad(e);
    return _poly_evald.get(e_rad.get_degree(), e_rad.get_order())[q1];
  }

 private:
  // angular element type
  typedef typename std::tuple_element<0, typename ELEM::container_t>::type ang_t;
  // radial element type
  typedef typename std::tuple_element<1, typename ELEM::container_t>::type rad_t;

 private:
  const POLY_T &_poly_evald;

  typename ELEM::Acc::template get<ang_t> get_ang;
  typename ELEM::Acc::template get<rad_t> get_rad;
};

}  // end namespace boltzmann
