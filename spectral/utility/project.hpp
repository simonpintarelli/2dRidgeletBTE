#pragma once

// own includes -----------------------------------------------------------
#include "aux/timer.hpp"
#include "quadrature/quadrature_handler.hpp"
#include "spectral/basis/spectral_elem_accessor.hpp"
#include "spectral/basis/spectral_function.hpp"

// system includes --------------------------------------------------------
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <boost/lexical_cast.hpp>
#include <boost/multi_array.hpp>
#include <complex>
#include <fstream>
#include <iostream>
#include <string>


namespace boltzmann {
/**
 * @brief approximate f(phi, r) in polar basis
 *
 * @param dst          coefficient array
 * @param basis        basis
 * @param f            function without weight
 * @param exponent     e^{-r^2 * exponent}
 * @param weight       inner product weight
 * @param mass_matrix  mass matrix
 */
template <typename BASIS>
void
spectral_project(double* dst, /* output */
                 const BASIS& basis,
                 const std::function<double(double, double)>& f,
                 const double exponent,
                 const double weight,
                 const Eigen::SparseMatrix<double>& mass_matrix)
{
  boltzmann::Timer timer;
  typedef typename BASIS::elem_t elem_t;
  typedef typename std::tuple_element<1, typename BASIS::elem_t::container_t>::type radial_elem_t;
  typedef TensorProductQuadrature<QMidpoint, QMaxwell> quad_t;
  typedef std::shared_ptr<quad_t> qptr_t;
  QuadratureHandler<> quad_handler;
  const int N = basis.n_dofs();
  typedef Eigen::VectorXd vec_t;
  vec_t rhs(N);
  // Assemble rhs
  timer.restart();
  for (auto it = basis.begin(); it != basis.end(); ++it) {
    const auto i = it - basis.begin();
    typename elem_t::Acc::template get<radial_elem_t> getter;
    const auto& phi = getter(*it);
    // const double beta = phi.get_beta();
    const double alpha = exponent + weight + phi.w();
    const auto& quad =
        quad_handler.get_quad(typename QuadratureHandler<>::descriptor(120, 100, alpha));

    double val = 0;
    for (unsigned int q = 0; q < quad.size(); ++q) {
      val += it->evaluate(quad.pts(q)[0], quad.pts(q)[1]) * f(quad.pts(q)[0], quad.pts(q)[1]) *
             quad.wts(q);
    }
    rhs[i] = val;
  }
  typedef Eigen::Map<vec_t> vec_wrapper_t;
  vec_wrapper_t x(dst, N);

  timer.restart();
  Eigen::SparseLU<Eigen::SparseMatrix<double> > lu;
  lu.compute(mass_matrix);
  x = lu.solve(rhs);
}

}  // end namespace boltzmann
