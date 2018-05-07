#pragma once

#include <algorithm>

#include "spectral/basis/basis_types.hpp"

#include "eval_handlers/hermite_eval_handler_2d.hpp"
#include "spectral/hermitenw.hpp"

namespace boltzmann {

namespace _local {

template <typename e>
using POLY_DEFAULT = HermiteNW<e>;
}  // _local

template <typename NUMERIC = double, template <class T> class POLY = _local::POLY_DEFAULT>
struct hermite_evaluator2d
{
 private:
  typedef hermite_basis_t::elem_t elem_t;

 public:
  /**
   *  @brief create Hermite polynomial evaluator in 2d
   *
   *  Detailed description
   *
   *  @param basis Hermite basis
   *  @param x     evaluation points
   *  @return Hermite evaluator handler of type @a HermiteEvalHandler2d
   */
  static HermiteEvalHandler2d<POLY<NUMERIC>, elem_t> make(const hermite_basis_t &basis,
                                                          std::vector<NUMERIC> x)
  {
    typedef NUMERIC numeric_t;

    typedef typename hermite_basis_t::elem_t elem_t;
    typename elem_t::Acc::template get<HermiteHX_t> getX;
    typename elem_t::Acc::template get<HermiteHY_t> getY;

    typedef POLY<NUMERIC> poly_t;
    auto hx_max = std::max_element(
        basis.begin(), basis.end(), [&getX, &getY](const elem_t &a, const elem_t &b) {
          return getX(a).get_degree() < getX(b).get_degree();
        });

    auto hy_max = std::max_element(
        basis.begin(), basis.end(), [&getX, &getY](const elem_t &a, const elem_t &b) {
          return getY(a).get_degree() < getY(b).get_degree();
        });

    int K = std::max(getX(*hx_max).get_degree(), getY(*hy_max).get_degree()) + 1;
    // debug output
    std::cout << "max degree: " << K << "\n";
    std::unique_ptr<poly_t> H = std::make_unique<poly_t>(K);
    H->compute(x);

    return HermiteEvalHandler2d<poly_t, elem_t>(std::move(H));
  }
};

}  // boltzmann
