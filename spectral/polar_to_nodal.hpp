#pragma once

#include "base/array_buffer.hpp"
#include "base/exceptions.hpp"
#include "spectral/basis/spectral_basis_factory_hermite.hpp"
#include "spectral/basis/spectral_basis_factory_ks.hpp"
#include "spectral/basis/toolbox/spectral_basis.hpp"

#include "hermite_to_nodal.hpp"
#include "polar_to_hermite.hpp"

#include <omp.h>
#include <Eigen/Dense>
#include <cassert>
#include <memory>
#include <string>


namespace boltzmann {

template <typename PolarBasis = SpectralBasisFactoryKS::basis_type>
class Polar2Nodal
{
 public:
  typedef double numeric_t;
  typedef PolarBasis polar_basis_t;
  typedef SpectralBasisFactoryHN::basis_type hermite_basis_t;
  typedef Polar2Hermite<polar_basis_t, hermite_basis_t> p2h_t;
  typedef Hermite2Nodal<hermite_basis_t> h2n_t;

 public:
  Polar2Nodal() { /* empty */}

  /**
   *  @brief Transform coefficients btw Polar-Laguerre and nodal basis. The
   *         nodes of the nodal basis are located at the underlying
   *         Gauss-Hermite quadrature points.
   *
   *  @param a exp weight factor in quad rule, e.g. for basis functions which
   *           decay like \f$exp(-r^/2)\f$, @param a is 1.0
   */
  Polar2Nodal(const polar_basis_t &polar_basis, double a = 1.0);

 public:
  void init(const polar_basis_t &polar_basis, double a);
  std::shared_ptr<p2h_t> get_p2h() const { return p2h_; }
  std::shared_ptr<h2n_t> get_h2n() const { return h2n_; }
  /// size Polar-Laguerre basis
  int N() const { return N_; }
  /// max. polynomial degree
  int K() const { return K_; }

  // void to_nodal(numeric_t* dst, const numeric_t* src, bool transpose) const;
  // void to_polar(numeric_t* dst, const numeric_t* src, bool transpose) const;

  template <typename DERIVED1, typename DERIVED2>
  void to_nodal(Eigen::DenseBase<DERIVED1> &dst,
                const Eigen::DenseBase<DERIVED2> &src,
                bool transpose = false) const;

  template <typename DERIVED1, typename DERIVED2>
  void to_polar(Eigen::DenseBase<DERIVED1> &dst,
                const Eigen::DenseBase<DERIVED2> &src,
                bool transpose = false) const;

 private:
  std::shared_ptr<p2h_t> p2h_;
  std::shared_ptr<h2n_t> h2n_;
  int N_;
  int K_;
  bool is_initialized_ = false;

  thread_local static ArrayBuffer<> buf_;
};

template <typename PolarBasis>
thread_local ArrayBuffer<> Polar2Nodal<PolarBasis>::buf_;

template <typename PolarBasis>
Polar2Nodal<PolarBasis>::Polar2Nodal(const polar_basis_t &polar_basis, double a)
{
  this->init(polar_basis, a);
}

//  -------------------------------------------------------------------------------------
template <typename PolarBasis>
void
Polar2Nodal<PolarBasis>::init(const polar_basis_t &polar_basis, double a)
{
  K_ = spectral::get_max_k(polar_basis) + 1;
  N_ = polar_basis.n_dofs();

  hermite_basis_t hermite_basis;
  SpectralBasisFactoryHN::create(hermite_basis, K_, 2);

  ASSERT(hermite_basis.n_dofs() == polar_basis.n_dofs());

  // make_unique not available in c++11
  // p2h_ = std::make_unique<p2h_t>(polar_basis, hermite_basis);
  // h2n_ = std::make_unique<h2n_t>(hermite_basis, K);
  typedef Eigen::MatrixXd mat_t;
  p2h_ = std::make_shared<p2h_t>(polar_basis, hermite_basis);
  const int K = K_;
  if (a == 1.0) {
    h2n_ = std::make_shared<h2n_t>(
        hermite_basis, K_, [K](mat_t &m1, mat_t &m2) { H2N_1d<>::create(m1, m2, K); });
  } else {
    h2n_ = std::make_shared<h2n_t>(
        hermite_basis, K_, [K, a](mat_t &m1, mat_t &m2) { H2NG_1d::create(m1, m2, K, a); });
  }

  is_initialized_ = true;
}

template <typename PolarBasis>
template <typename DERIVED1, typename DERIVED2>
void
Polar2Nodal<PolarBasis>::to_nodal(Eigen::DenseBase<DERIVED1> &dst,
                                  const Eigen::DenseBase<DERIVED2> &src,
                                  bool transpose) const
{
  auto buffer = buf_.get<Eigen::VectorXd>(N_);
  BOOST_ASSERT(is_initialized_);

  if (!transpose) {
    p2h_->to_hermite(buffer, src);
    h2n_->to_nodal(dst, buffer);
  } else {
    throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) +
                             " not implemented");
  }
}

template <typename PolarBasis>
template <typename DERIVED1, typename DERIVED2>
void
Polar2Nodal<PolarBasis>::to_polar(Eigen::DenseBase<DERIVED1> &dst,
                                  const Eigen::DenseBase<DERIVED2> &src,
                                  bool transpose) const
{
  BOOST_ASSERT(is_initialized_);
  auto buffer = buf_.get<Eigen::VectorXd>(N_);

  if (!transpose) {
    h2n_->to_hermite(buffer, src);
    p2h_->to_polar(dst, buffer);
  } else {
    throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) +
                             " not implemented");
  }
}

}  // end namespace boltzmann
