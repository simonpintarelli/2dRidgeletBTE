#pragma once

// system includes ------------------------------------------------------------
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <memory>
#include <iostream>
#include <vector>

#include "base/exceptions.hpp"
#include "spectral/basis/spectral_basis_factory_ks.hpp"


namespace boltzmann {
class CollisionTensorGalerkin
{
 private:
  typedef Eigen::SparseMatrix<double> sparse_matrix_t;
  typedef std::shared_ptr<sparse_matrix_t> ptr_t;
  typedef Eigen::SparseLU<sparse_matrix_t> lu_t;
  typedef typename SpectralBasisFactoryKS::basis_type basis_t;

  typedef Eigen::VectorXd vec_t;
  typedef Eigen::MatrixXd matrix_t;
  typedef Eigen::Matrix4d m4_t;
  typedef Eigen::DiagonalMatrix<double, -1> diag_t;

 public:
  CollisionTensorGalerkin(const basis_t &basis);
  CollisionTensorGalerkin()
      : N_(0)
  {
  }

  void add(ptr_t &slice, unsigned int j);
  void apply(double *out, const double *in) const;
  void apply(double *out, const double *in, const unsigned int L) const;

  void apply_adaptive(double *out, const double *in, int nmax) const;

  void project(double *out, const double *in) const;

  const sparse_matrix_t &get(int j) const;

  /**
   * @brief read tensor from file
   *
   * @param fname Filename
   */
  void read_hdf5(const char *fname);

 private:
  /// basis size
  unsigned int N_;
  /// tensor entries
  std::vector<ptr_t> slices_;

  matrix_t Ht_;
  m4_t HtHinv_;
  diag_t Sinv_;
};

// ------------------------------------------------------------
inline void
CollisionTensorGalerkin::apply(double *out, const double *in) const
{
  typedef Eigen::Map<const vec_t> cvec_t;  // constant vector
  typedef Eigen::Map<vec_t> mvec_t;        // mutable vector

  mvec_t vout(out, N_);
  cvec_t vin(in, N_);

#pragma omp parallel for
  for (unsigned int i = 0; i < N_; ++i) {
    vout(i) = vin.dot((*slices_[i]) * vin);
  }

  vout = Sinv_ * vout;
}

// ------------------------------------------------------------
inline void
CollisionTensorGalerkin::project(double *out, const double *in) const
{
  typedef Eigen::Map<const vec_t> cvec_t;  // constant vector
  typedef Eigen::Map<vec_t> mvec_t;        // mutable vector

  cvec_t vin(in, N_);
  mvec_t vout(out, N_);

  Eigen::Vector4d lambda = HtHinv_ * Ht_ * (vout - vin);
  vout -= Sinv_ * Ht_.transpose() * lambda;
}

// ------------------------------------------------------------
inline void
CollisionTensorGalerkin::apply_adaptive(double *out, const double *in, int nmax) const
{
  typedef Eigen::Map<const vec_t> cvec_t;  // constant vector
  typedef Eigen::Map<vec_t> mvec_t;        // mutable vector

  mvec_t vout(out, N_);
  cvec_t vin(in, nmax);

#pragma omp parallel for
  for (unsigned int i = 0; i < N_; ++i) {
    vout(i) = vin.dot((*slices_[i]).topLeftCorner(nmax, nmax) * vin);
  }
  vout = Sinv_ * vout;
}

// ----------------------------------------------------------------------
inline void
CollisionTensorGalerkin::apply(double *out, const double *in, const unsigned int L) const
{
  typedef Eigen::Map<Eigen::VectorXd> vec_t;
  typedef Eigen::Map<const Eigen::VectorXd> const_vec_t;

  for (unsigned int j = 0; j < N_; j++) {
    for (unsigned int l = 0; l < L; ++l) {
      const_vec_t vin(in + N_ * l, N_);
      out[N_ * l + j] = vin.dot(*slices_[j] * vin);
    }
  }

  for (unsigned int l = 0; l < L; ++l) {
    vec_t vout(out + N_ * l, N_);
    vout = Sinv_ * vout;
  }
}

}  // end boltzmann
