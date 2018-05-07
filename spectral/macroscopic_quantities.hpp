#pragma once

#include <Eigen/Dense>
#include <boost/assert.hpp>
#include <type_traits>

#include "spectral/basis/spectral_basis_factory_hermite.hpp"
#include "spectral/basis/spectral_basis_factory_ks.hpp"
#include "spectral/basis/toolbox/spectral_basis.hpp"
#include "spectral/polar_to_hermite.hpp"


namespace boltzmann {

class MQEval;  // forward declaration

namespace detail_ {
/**
 * @helper to evaluate macroscopic quantities
 *
 * @param mq
 *
 * @return
 */
struct MQEval_helper
{
 public:
  typedef Eigen::Vector3d vector_t;
  typedef Eigen::Matrix3d tensor_t;
  typedef double scalar_t;

 public:
  MQEval_helper(const MQEval &mq)
      : mq_(mq)
  { /* empty */
  }

  void operator()(const double *ptr_c, unsigned int N);

  template <typename DERIVED>
  void operator()(const Eigen::DenseBase<DERIVED> &c);

 public:
  double m;    // mass
  double e;    // energy
  vector_t v;  // velocity
  vector_t r;  // energy flow
  tensor_t P;  // pressure
  vector_t q;  // heat flow
  tensor_t M;  // momentum flow

 private:
  const MQEval &mq_;
};

}  // detail_

/**
 * @brief macroscopic quantities evaluator
 *
 */
class MQEval
{
 public:
  typedef detail_::MQEval_helper evaluator_t;

 public:
  MQEval() { /*  default constructor */}

  template <typename BASIS>
  MQEval(const BASIS &basis)
  {
    init(basis);
  }

  /**
   *  @brief evaluator to do actual compuations
   *
   *  Returns a struct storing the moments, it has a const reference to *this
   * (where the
   *  coefficients are stored).
   *
   *  Attention: MQEval_helper depends on MQEval by a reference, make sure it
   * does not run out of
   * scope as
   *  long as the MQEval_helper is in use.
   *
   */
  detail_::MQEval_helper evaluator() const { return detail_::MQEval_helper(*this); }

  /**
   *
   * @param basis  Polar-Laguerre basis
   */
  template <typename BASIS>
  void init(const BASIS &basis);

 private:
  typedef Eigen::VectorXd vec_t;
  typedef Eigen::ArrayXd array_t;
  typedef Eigen::MatrixXd mat_t;

 private:
  /// basis size
  unsigned int N_ = 0;
  const double tol_ = 1e-10;

  array_t mass_;
  array_t energy_;
  array_t ux_;
  array_t uy_;
  /// momentum flow
  array_t uxx_;
  array_t uyy_;
  array_t uxy_;
  /// energy flow
  array_t rx_;
  array_t ry_;

 public:
  //@{
  /// coefficients in Polar-Laguerre basis
  const array_t &cmass() const { return mass_; }
  const array_t &cenergy() const { return energy_; }
  const array_t &cux() const { return ux_; }
  const array_t &cuy() const { return uy_; }
  const array_t &cuxx() const { return uxx_; }
  const array_t &cuyy() const { return uyy_; }
  const array_t &cuxy() const { return uxy_; }
  const array_t &crx() const { return rx_; }
  const array_t &cry() const { return ry_; }
  //@}

  unsigned int N() const { return N_; }

 private:
  /**
   * @brief trim non zero coefficients starting from the end
   */
  template <typename DERIVED>
  void trim_coeff(Eigen::DenseBase<DERIVED> &coeffs);
};

// ---------------------------------------------------------------------------
template <typename BASIS>
void
MQEval::init(const BASIS &basis)
{
  typedef BASIS polar_basis_t;
  // compute coefficients

  int K = spectral::get_max_k(basis) + 1;
  int N = basis.n_dofs();
  N_ = N;

  typedef SpectralBasisFactoryHN::basis_type hermite_basis_t;
  hermite_basis_t hermite_basis;

  SpectralBasisFactoryHN::create(hermite_basis, K, 2);
  Polar2Hermite<polar_basis_t, hermite_basis_t> P2H(basis, hermite_basis);
  typedef typename hermite_basis_t::elem_t hermite_elem_t;

  typedef typename boost::mpl::at_c<typename hermite_elem_t::types_t, 0>::type hx_t;
  typedef typename boost::mpl::at_c<typename hermite_elem_t::types_t, 1>::type hy_t;

  typename hermite_elem_t::Acc::template get<hx_t> get_hx;
  typename hermite_elem_t::Acc::template get<hy_t> get_hy;

  // Hermite quadrature
  QHermiteW quad(0.5, K);
  // Hermite polynomials
  HermiteNW<double> hermw(K);
  hermw.compute(quad.pts());

  // test for mass
  vec_t herm_coeffs(N);
  // apply "quadrature":
  Eigen::Map<const array_t> x(quad.points_data(), K);
  Eigen::Map<const array_t> w(quad.weights_data(), K);

  // ------------------------------
  // MASS
  for (int i = 0; i < N; ++i) {
    int kx = get_hx(hermite_basis.get_elem(i)).get_id().k;
    int ky = get_hy(hermite_basis.get_elem(i)).get_id().k;
    auto hx = hermw.get_array(kx);
    auto hy = hermw.get_array(ky);

    herm_coeffs[i] = (hx * w).sum() * (hy * w).sum();
  }
  mass_.resize(N);
  P2H.to_hermite_T(mass_, herm_coeffs);

  // ------------------------------
  // ENERGY
  for (int i = 0; i < N; ++i) {
    int kx = get_hx(hermite_basis.get_elem(i)).get_id().k;
    int ky = get_hy(hermite_basis.get_elem(i)).get_id().k;

    auto hx = hermw.get_array(kx);
    auto hy = hermw.get_array(ky);

    herm_coeffs[i] =
        (x * x * hx * w).sum() * (hy * w).sum() + (x * x * hy * w).sum() * (hx * w).sum();
  }
  energy_.resize(N);
  P2H.to_hermite_T(energy_, herm_coeffs);

  // ------------------------------
  // MOMENTUM
  vec_t herm_coeffs2(N);
  for (int i = 0; i < N; ++i) {
    int kx = get_hx(hermite_basis.get_elem(i)).get_id().k;
    int ky = get_hy(hermite_basis.get_elem(i)).get_id().k;
    auto hx = hermw.get_array(kx);
    auto hy = hermw.get_array(ky);
    double sumx = (x * hx * w).sum() * (hy * w).sum();
    double sumy = (x * hy * w).sum() * (hx * w).sum();
    herm_coeffs[i] = sumx;
    herm_coeffs2[i] = sumy;
  }

  ux_.resize(N);
  P2H.to_hermite_T(ux_, herm_coeffs);
  uy_.resize(N);
  P2H.to_hermite_T(uy_, herm_coeffs2);

  // resize to non-zero contribution
  trim_coeff(mass_);
  trim_coeff(energy_);
  trim_coeff(ux_);
  trim_coeff(uy_);

  uxx_.resize(N);
  uxy_.resize(N);
  uyy_.resize(N);
  // ------------------------------
  // Momentum flow
  for (int i = 0; i < N; ++i) {
    int kx = get_hx(hermite_basis.get_elem(i)).get_id().k;
    int ky = get_hy(hermite_basis.get_elem(i)).get_id().k;
    auto hx = hermw.get_array(kx);
    auto hy = hermw.get_array(ky);
    const double tx = (x * hx * w).sum();
    const double ty = (x * hy * w).sum();
    uxx_[i] = (x * x * hx * w).sum() * (hy * w).sum();
    uxy_[i] = (x * hx * w).sum() * (x * hy * w).sum();
    uyy_[i] = (x * x * hy * w).sum() * (hx * w).sum();
  }
  herm_coeffs = uxx_;
  P2H.to_hermite_T(uxx_, herm_coeffs);
  herm_coeffs = uxy_;
  P2H.to_hermite_T(uxy_, herm_coeffs);
  herm_coeffs = uyy_;
  P2H.to_hermite_T(uyy_, herm_coeffs);

  // ------------------------------
  // Momentum flow
  uxx_.resize(N);
  uxy_.resize(N);
  uyy_.resize(N);
  for (int i = 0; i < N; ++i) {
    int kx = get_hx(hermite_basis.get_elem(i)).get_id().k;
    int ky = get_hy(hermite_basis.get_elem(i)).get_id().k;
    auto hx = hermw.get_array(kx);
    auto hy = hermw.get_array(ky);

    uxx_[i] = (x * x * hx * w).sum() * (hy * w).sum();
    uxy_[i] = (x * hx * w).sum() * (x * hy * w).sum();
    uyy_[i] = (x * x * hy * w).sum() * (hx * w).sum();
  }
  herm_coeffs = uxx_;
  P2H.to_hermite_T(uxx_, herm_coeffs);
  herm_coeffs = uxy_;
  P2H.to_hermite_T(uxy_, herm_coeffs);
  herm_coeffs = uyy_;
  P2H.to_hermite_T(uyy_, herm_coeffs);

  // ------------------------------
  // Energy flow
  rx_.resize(N);
  ry_.resize(N);
  for (int i = 0; i < N; ++i) {
    int kx = get_hx(hermite_basis.get_elem(i)).get_id().k;
    int ky = get_hy(hermite_basis.get_elem(i)).get_id().k;
    auto hx = hermw.get_array(kx);
    auto hy = hermw.get_array(ky);

    rx_[i] =
        (x * x * x * hx * w).sum() * (hy * w).sum() + (x * hx * w).sum() * (x * x * hy * w).sum();
    ry_[i] =
        (x * x * x * hy * w).sum() * (hx * w).sum() + (x * hy * w).sum() * (x * x * hx * w).sum();
  }
  herm_coeffs = rx_;
  P2H.to_hermite_T(rx_, herm_coeffs);
  herm_coeffs = ry_;
  P2H.to_hermite_T(ry_, herm_coeffs);

  trim_coeff(uxx_);
  trim_coeff(uxy_);
  trim_coeff(uyy_);

  trim_coeff(rx_);
  trim_coeff(ry_);
}

template <typename DERIVED>
void
MQEval::trim_coeff(Eigen::DenseBase<DERIVED> &coeffs)
{
  for (unsigned int i = N_ - 1; i >= 0; --i) {
    if (std::abs(coeffs[i]) > tol_) {
      DERIVED tmp = coeffs.segment(0, i + 1);
      coeffs.derived().resize(i + 1);
      coeffs = tmp;
      break;
    }
  }
}

namespace detail_ {
template <typename DERIVED>
void
MQEval_helper::operator()(const Eigen::DenseBase<DERIVED> &c)
{
  static_assert(DERIVED::RowsAtCompileTime == 1 || DERIVED::ColsAtCompileTime == 1,
                "Shape mismatch");

  BOOST_ASSERT(c.cols() * c.rows() == mq_.N());
  auto &cmass = mq_.cmass();
  auto &cux = mq_.cux();
  auto &cuy = mq_.cuy();
  auto &ce = mq_.cenergy();
  auto &cuxx = mq_.cuxx();
  auto &cuxy = mq_.cuxy();
  auto &cuyy = mq_.cuyy();
  auto &crx = mq_.crx();
  auto &cry = mq_.cry();

  auto ca = c.derived().array();

  double rho = (ca.segment(0, cmass.size()) * cmass).sum();
  m = rho;
  double rho_vx = (ca.segment(0, cux.size()) * cux).sum();
  double rho_vy = (ca.segment(0, cuy.size()) * cuy).sum();
  // total energy
  double rho_e = (ca.segment(0, ce.size()) * ce).sum();
  e = rho_e / rho;
  // energy density per unit volume
  double w = 0.5 * rho_e;
  double mxx = (ca.segment(0, cuxx.size()) * cuxx).sum();
  double mxy = (ca.segment(0, cuxy.size()) * cuxy).sum();
  double myy = (ca.segment(0, cuyy.size()) * cuyy).sum();
  M << mxx, mxy, 0, mxy, myy, 0, 0, 0, 0;
  v(0) = rho_vx / rho;
  v(1) = rho_vy / rho;
  v(2) = 0;
  double pxx = mxx - rho * v(0) * v(0);
  double pxy = mxy - rho * v(1) * v(0);
  double pyy = myy - rho * v(1) * v(1);
  P << pxx, pxy, 0, pxy, pyy, 0, 0, 0, 0;
  // energy flow
  r(0) = (ca.segment(0, crx.size()) * crx).sum();
  r(1) = (ca.segment(0, cry.size()) * cry).sum();
  r(2) = 0;
  double v2 = v.squaredNorm();
  // heat flow
  q(0) = 0.5 * (r(0) - 2 * (v * M.row(0)).sum() + rho * v2 + 2 * v(0) * w + v2 * v(0) * rho);
  q(1) = 0.5 * (r(1) - 2 * (v * M.row(1)).sum() + rho * v2 + 2 * v(1) * w + v2 * v(1) * rho);
  q(2) = 0;
}

inline void
MQEval_helper::operator()(const double *ptr_c, unsigned int N)
{
  assert(N == mq_.N());
  Eigen::Map<const Eigen::ArrayXd> c(ptr_c, N);
  this->operator()(c);
}

}  // detail_

}  // end namespace boltzmann
