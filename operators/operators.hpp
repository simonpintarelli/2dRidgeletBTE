#pragma once

// system includes --------------------------------------------------
#include <Eigen/Dense>

// own includes -----------------------------------------------------
#include <ridgelet/ridgelet_cell_array.hpp>
#include <ridgelet/rt.hpp>


template <typename FRAME>
std::vector<double>
make_inv_diagonal_preconditioner(const FRAME &frame, double vx, double vy)
{
  auto &lambdas = frame.lambdas();
  unsigned int N = lambdas.size();
  std::vector<double> d(N);

  Eigen::Vector2d sjk;
  Eigen::Vector2d v;
  v(0) = vx;
  v(1) = vy;

  for (unsigned int i = 0; i < N; ++i) {
    int j = lambdas[i].j;
    double p2mn = std::pow(2., j - 1);
    int k = lambdas[i].k;
    if (lambdas[i].t == rt_type::S) {
      d[i] = 1.0;
    } else if (lambdas[i].t == rt_type::X) {
      sjk(0) = 1;
      sjk(1) = double(k) / p2mn;
      sjk.normalize();
      d[i] = 1 + std::pow(2., j) * std::abs(sjk.dot(v));
    } else if (lambdas[i].t == rt_type::Y) {
      sjk(0) = double(k) / p2mn;
      sjk(1) = 1;
      sjk.normalize();
      d[i] = 1 + std::pow(2., j) * std::abs(sjk.dot(v));
    } else if (lambdas[i].t == rt_type::D) {
      k = (k < 0) ? -1 : 1;
      sjk(0) = 1;
      sjk(1) = k;
      sjk.normalize();
      d[i] = 1 + std::pow(2., j) * std::abs(sjk.dot(v));
    } else {
      assert(false);
    }
  }

  return d;
}

template <typename ARRAY_T>
class DiagonalOperator
{
 private:
  typedef RidgeletCellArray<ARRAY_T> rca_t;

 public:
  DiagonalOperator(const std::vector<double> &d)
      : d_(d)
  { /* empty  */
  }

  DiagonalOperator() {}

  DiagonalOperator &operator=(const DiagonalOperator &other) { d_ = other.d_; }

  DiagonalOperator(std::vector<double> &&d)
      : d_(std::forward<std::vector<double>>(d))
  { /* empty  */
  }

  void apply(rca_t &dst) const;
  void apply(rca_t &dst, const rca_t &src) const;

  void invert();

 private:
  std::vector<double> d_;
};

template <typename ARRAY_T>
void
DiagonalOperator<ARRAY_T>::apply(rca_t &dst) const
{
  assert(d_.size() == dst.coeffs().size());

  for (unsigned int i = 0; i < d_.size(); ++i) {
    dst[i] *= d_[i];
  }
}

template <typename ARRAY_T>
void
DiagonalOperator<ARRAY_T>::apply(rca_t &dst, const rca_t &src) const
{
  assert(d_.size() == dst.coeffs().size());
  assert(d_.size() == src.coeffs().size());

  for (unsigned int i = 0; i < d_.size(); ++i) {
    dst[i] = d_[i] * src[i];
  }
}

template <typename ARRAY_T>
void
DiagonalOperator<ARRAY_T>::invert()
{
  for (unsigned int i = 0; i < d_.size(); ++i) {
    d_[i] = 1. / d_[i];
  }
}

// ================================================================================
// ================================================================================
// ================================================================================
class TransportOperator
{
 protected:
  typedef Eigen::ArrayXXcd complex_array_t;
  typedef Eigen::ArrayXd col_t;  // (column vector)

 public:
  TransportOperator(double vx, double vy, double Lx, double Ly, int Nx, int Ny, double dt = 1.0)
      : vx_(vx)
      , vy_(vy)
      , Lx_(Lx)
      , Ly_(Ly)
      , Nx_(Nx)
      , Ny_(Ny)
      , dt_(dt)
  {
    Eigen::Vector2d v;
    v(0) = vx;
    v(1) = vy;

    // init xi_x
    if (Nx % 2 == 0)
      xi_x_ = col_t::LinSpaced(Nx, -Nx / 2, Nx / 2 - 1) / Lx;
    else
      xi_x_ = col_t::LinSpaced(Nx, -Nx / 2, Nx / 2) / Lx;

    // init xi_y
    if (Ny % 2 == 0)
      xi_y_ = col_t::LinSpaced(Ny, -Ny / 2, Ny / 2 - 1) / Ly;
    else
      xi_y_ = col_t::LinSpaced(Ny, -Ny / 2, Ny / 2) / Ly;
  }

  template <typename COMPLEX_ARRAY>
  void apply(COMPLEX_ARRAY &dst, const COMPLEX_ARRAY &src, bool conj = false) const;

  template <typename COMPLEX_ARRAY>
  void apply_bckwrd_euler(COMPLEX_ARRAY &dst) const;

 protected:
  double vx_;
  double vy_;
  double Lx_;
  double Ly_;
  unsigned int Nx_;
  unsigned int Ny_;

  col_t xi_x_;
  col_t xi_y_;
  const double twoPI = 2 * 3.141592653589793238462643;
  const std::complex<double> I = std::complex<double>(0, 1);
  double dt_;
};

template <typename COMPLEX_ARRAY>
void
TransportOperator::apply(COMPLEX_ARRAY &dst, const COMPLEX_ARRAY &src, bool conj) const
{
  typedef typename COMPLEX_ARRAY::Scalar numeric_t;

  const double f = conj ? -1 : 1;
  dst = src +
        f * (twoPI * dt_ * I) *
            (vx_ * xi_x_.transpose().replicate(Ny_, 1) + vy_ * xi_y_.replicate(1, Nx_))
                .cast<numeric_t>()
                .array() *
            src.array();
}

template <typename COMPLEX_ARRAY>
void
TransportOperator::apply_bckwrd_euler(COMPLEX_ARRAY &dst) const
{
  typedef typename COMPLEX_ARRAY::Scalar numeric_t;
  dst = dst.cwiseQuotient(
      COMPLEX_ARRAY::Ones(Ny_, Nx_) +
      twoPI * dt_ * I *
          (vx_ * xi_x_.transpose().replicate(Ny_, 1) + vy_ * xi_y_.replicate(1, Nx_))
              .cast<numeric_t>()
              .array());
}

// ================================================================================
/**
 *   @brief \f$ T' T  \f$
 */
class AhAOp : public TransportOperator
{
 private:
  typedef Eigen::ArrayXXcd complex_array_t;
  typedef Eigen::ArrayXd col_t;

 public:
  AhAOp(double vx, double vy, double Lx, double Ly, int Nx, int Ny, double dt = 1.0)
      : TransportOperator(vx, vy, Lx, Ly, Nx, Ny, dt)
  {
    dt2_ = dt_ * dt_;
  }

  template <typename DERIVED1, typename DERIVED2>
  void apply(Eigen::ArrayBase<DERIVED1> &dst, const Eigen::ArrayBase<DERIVED2> &src) const;

 private:
  using TransportOperator::xi_x_;
  using TransportOperator::xi_y_;
  double dt2_;
};

template <typename DERIVED1, typename DERIVED2>
void
AhAOp::apply(Eigen::ArrayBase<DERIVED1> &dst, const Eigen::ArrayBase<DERIVED2> &src) const
{
  static_assert(std::is_same<typename DERIVED1::Scalar, typename DERIVED2::Scalar>::value,
                "type mismatch");
  typedef typename DERIVED1::Scalar numeric_t;

  dst = src +
        twoPI * twoPI * dt2_ *
            (vx_ * xi_x_.transpose().replicate(Ny_, 1) + vy_ * xi_y_.replicate(1, Nx_))
                .cwiseAbs2()
                .cast<numeric_t>()
                .array() *
            src;
}

// ================================================================================
// ================================================================================
// ================================================================================
// transport operator with boundary conditions
class TransportOperatorBC : public TransportOperator
{
 public:
  template <typename DERIVED>
  TransportOperatorBC(const Eigen::DenseBase<DERIVED> &sigma,
                      double vx,
                      double vy,
                      double Lx,
                      double Ly,
                      int Nx,
                      int Ny,
                      double dt = 1.0)
      : TransportOperator(vx, vy, Lx, Ly, Nx, Ny, dt)
  {
    sigma_ = sigma;
  }

  template <typename COMPLEX_ARRAY>
  void apply(COMPLEX_ARRAY &dst, const COMPLEX_ARRAY &src, bool conj = false) const;

 private:
#ifdef USE_PLANNED_FFT
  typedef FFTr2c<PlannerR2C> fft_t;
#else
  typedef FFTr2c<PlannerR2COD> fft_t;
#endif

  typedef Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> array_t;

 private:
  Eigen::ArrayXXd sigma_;
};

template <typename COMPLEX_ARRAY>
void
TransportOperatorBC::apply(COMPLEX_ARRAY &dst, const COMPLEX_ARRAY &src, bool conj) const
{
  typedef COMPLEX_ARRAY complex_array_t;
  typedef typename complex_array_t::Scalar numeric_t;
  const double f = conj ? -1 : 1;

  // TODO: (inefficient) memory allocation
  array_t fn(src.rows(), src.cols());
  double r = std::sqrt(vx_ * vx_ + vy_ * vy_);
  fft_t fft;
  fft.ift(fn, src);
  fn *= sigma_ * r;
  fft.ft(dst, fn, false);
  dst *= dt_;
  dst += src +
         f * (twoPI * dt_ * I) *
             (vx_ * xi_x_.transpose().replicate(Ny_, 1) + vy_ * xi_y_.replicate(1, Nx_))
                 .cast<numeric_t>()
                 .array() *
             src.array();
}

/**
 * @brief \f$ T'T \f$  * where T is of type \a TransportOperatorBC an absorption
 * term (otherwise
 * identical to \a AhAOp)
 *
 *
 */
class AhAOpsigma : public TransportOperatorBC
{
 private:
  typedef TransportOperatorBC transport_op_t;

 public:
  template <typename DERIVED>
  AhAOpsigma(const Eigen::DenseBase<DERIVED> &sigma,
             double vx,
             double vy,
             double Lx,
             double Ly,
             int Nx,
             int Ny,
             double dt = 1.0)
      : TransportOperatorBC(sigma, vx, vy, Lx, Ly, Nx, Ny, dt)
  { /* empty  */
  }

  template <typename COMPLEX_ARRAY>
  void apply(COMPLEX_ARRAY &dst, const COMPLEX_ARRAY &src) const
  {
    typedef COMPLEX_ARRAY complex_array_t;
    typedef typename complex_array_t::Scalar numeric_t;

    complex_array_t tmp(dst.rows(), dst.cols());

    transport_op_t::apply(tmp, src);
    transport_op_t::apply(dst, tmp, true);
  }
};

// ================================================================================
// ================================================================================
// ================================================================================
/// Preconditioned transport operator \f$ D^{-1} T D\f$
template <typename RT_TYPE, typename OP_T>
class PTransportOp_Base
{
 private:
  typedef RT_TYPE rt_t;
  typedef typename rt_t::complex_array_t complex_array_t;
  typedef OP_T op_t;
  typedef typename rt_t::rt_coeff_t rt_coeff_t;

 public:
  /**
   *
   * @param rt    ridgelet transform object
   * @param aha   A^T A
   * @param vx    velocity in x direction
   * @param vy    velocity in y direction
   */
  PTransportOp_Base(const rt_t &rt, const op_t &aha, double vx, double vy)
      : rt_(rt)
      , aha_(aha)
      , D_(make_inv_diagonal_preconditioner(rt.frame(), vx, vy))
      , rt_coeffs_(rt.frame())
  {
    fi.resize(rt.frame().Ny(), rt.frame().Nx());
    fo.resize(rt.frame().Ny(), rt.frame().Nx());

    // prepare preconditioner
    D_.invert();
  }

  void apply(RidgeletCellArray<rt_coeff_t> &dst, const RidgeletCellArray<rt_coeff_t> &src) const
  {
    // apply D_
    D_.apply(rt_coeffs_, src);
    rt_.irt(fi, rt_coeffs_.coeffs());
    assert(!fi.hasNaN());
    aha_.apply(fo, fi);
    assert(!fo.hasNaN());
    rt_.rt(dst.coeffs(), fo);
    D_.apply(dst);
  }

 private:
  const rt_t &rt_;
  const op_t &aha_;
  DiagonalOperator<rt_coeff_t> D_;

  mutable RidgeletCellArray<rt_coeff_t> rt_coeffs_;
  mutable complex_array_t fi;
  mutable complex_array_t fo;
};

template <typename RT_TYPE = RT<>>
using PTransportOp = PTransportOp_Base<RT_TYPE, AhAOp>;

template <typename RT_TYPE = RT<>>
using PTransportOpBC = PTransportOp_Base<RT_TYPE, AhAOpsigma>;

// ================================================================================
/// Preconditioned transport operator \f$ D^{-1} T D\f$
template <typename RT_TYPE, typename OP_T>
class PTransportOp_BaseId
{
 private:
  typedef RT_TYPE rt_t;
  typedef typename rt_t::complex_array_t complex_array_t;
  typedef OP_T op_t;
  typedef typename rt_t::rt_coeff_t rt_coeff_t;

 public:
  /**
   *
   * @param rt    ridgelet transform object
   * @param aha   A^T A
   * @param vx    velocity in x direction
   * @param vy    velocity in y direction
   */
  PTransportOp_BaseId(const rt_t &rt, const op_t &aha, double vx, double vy)
      : rt_(rt)
      , aha_(aha)
  {
    fi.resize(rt.frame().Ny(), rt.frame().Nx());
    fo.resize(rt.frame().Ny(), rt.frame().Nx());
  }

  void apply(RidgeletCellArray<rt_coeff_t> &dst, const RidgeletCellArray<rt_coeff_t> &src) const
  {
    rt_.irt(fi, src.coeffs());
    assert(!fi.hasNaN());
    aha_.apply(fo, fi);
    assert(!fo.hasNaN());
    rt_.rt(dst.coeffs(), fo);
  }

 private:
  const rt_t &rt_;
  const op_t &aha_;
  mutable complex_array_t fi;
  mutable complex_array_t fo;
};

template <typename RT_TYPE = RT<>>
using PTransportOpId = PTransportOp_BaseId<RT_TYPE, AhAOp>;
