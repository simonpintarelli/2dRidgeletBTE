#pragma once

// system includes ----------------------------------------
#include <unordered_map>
#include <vector>
// own includes -------------------------------------------
#include <base/hash_specializations.hpp>
#include <base/types.hpp>
#include "construction/ft.hpp"
#include "lambda.hpp"


/**
 *
 * @brief container to store Ridgelet Frame Fourier Coefficients
 *
 */
class RidgeletFrame
{
 public:
  typedef Eigen::SparseMatrix<double, Eigen::RowMajor> sparse_matrix_t;
  typedef Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> matrix_t;
  // typedef Eigen::ArrayXXd matrix_t;

  template <typename RIDGE_FT = ridge_ft<>>
  RidgeletFrame(unsigned int Jx,
                unsigned int Jy,
                unsigned int rho_x,
                unsigned int rho_y,
                RIDGE_FT _ = ridge_ft<>());

  RidgeletFrame() {}

 public:
  /**
   * @brief get X,Y,D Ridgelet Fourier coefficients
   *
   * @param lambda
   *
   * @return sparse matrix
   */
  const sparse_matrix_t &get_sparse(const lambda_t &lambda) const;

  /**
   * @brief get S Ridgelet Fourier coefficients
   *
   * @param lambda
   *
   * @return dense array
   */
  const matrix_t &get_dense(const lambda_t &lambda) const;

  const std::vector<lambda_t> &lambdas() const;
  unsigned int size() const;

  unsigned int rho_x() const;
  unsigned int rho_y() const;
  unsigned int Jx() const;
  unsigned int Jy() const;
  /// physical resolution in x-direction, #rows
  unsigned int Nx() const;
  /// physical resolution in y-direction, #cols
  unsigned int Ny() const;
  bool is_element(const lambda_t &lambda) const;
  double rf_norm(const lambda_t &lambda) const;

 private:
  sparse_matrix_t &insert_sparse(const lambda_t &lambda);
  matrix_t &insert_dense(const lambda_t &lambda);

 private:
  unsigned int Jx_;
  unsigned int Jy_;
  unsigned int rho_x_;
  unsigned int rho_y_;
  unsigned int Nx_;
  unsigned int Ny_;

  enum dense_sparse
  {
    DENSE,
    SPARSE
  };
  typedef std::unordered_map<lambda_t, matrix_t> dense_map_coeff_t;
  typedef std::unordered_map<lambda_t, sparse_matrix_t> sparse_map_coeff_t;

  std::shared_ptr<dense_map_coeff_t> dense_coeffs_ptr_;
  std::shared_ptr<sparse_map_coeff_t> sparse_coeffs_ptr_;
  std::shared_ptr<std::vector<lambda_t>> lambdas_ptr_;
};

// --------------------------------------------------------------------------------
template <typename RIDGE_FT>
RidgeletFrame::RidgeletFrame(
    unsigned int Jx, unsigned int Jy, unsigned int rho_x, unsigned int rho_y, RIDGE_FT _)
    : Jx_(Jx)
    , Jy_(Jy)
    , rho_x_(rho_x)
    , rho_y_(rho_y)
    , Nx_(0)
    , Ny_(0)
{
  // initialize objects
  dense_coeffs_ptr_ = std::make_shared<dense_map_coeff_t>();
  sparse_coeffs_ptr_ = std::make_shared<sparse_map_coeff_t>();
  lambdas_ptr_ = std::make_shared<std::vector<lambda_t>>();

  lambda_t lam(0, rt_type::S, 0);
  auto &mat = insert_dense(lam);
  RIDGE_FT::s_ridgelet_ft(mat, rho_x, rho_y);
  // also insert as sparse
  auto &sp_mat = insert_sparse(lam);
  sp_mat = mat.matrix().sparseView();

  // ------------------------------
  // x-ridgelets
  for (unsigned int j = 1; j <= Jx; ++j) {
    int kmax = std::pow(2, j - 1);
    for (int k = -kmax + 1; k < kmax; ++k) {
      lambda_t lam(j, rt_type::X, k);
      auto &mat = insert_sparse(lam);
      RIDGE_FT::x_ridgelet_ft(mat, j, k, rho_x, rho_y);
    }
  }
  // ------------------------------
  // y-ridgelets
  for (unsigned int j = 1; j <= Jy; ++j) {
    int kmax = std::pow(2, j - 1);
    for (int k = -kmax + 1; k < kmax; ++k) {
      lambda_t lam(j, rt_type::Y, k);
      auto &mat = insert_sparse(lam);
      RIDGE_FT::y_ridgelet_ft(mat, j, k, rho_x, rho_y);
    }
  }

  // ------------------------------
  // diagonal ridgelets
  for (unsigned int j = 1; j <= std::min(Jx, Jy) + (Jx != Jy); ++j) {
    // k = 2^(j-1)
    int kp = std::pow(2, j - 1);
    sparse_matrix_t ft_xp, ft_yp;
    RIDGE_FT::x_ridgelet_ft(ft_xp, j, kp, rho_x, rho_y);
    RIDGE_FT::y_ridgelet_ft(ft_yp, j, kp, rho_x, rho_y);

    auto &matp = insert_sparse(lambda_t(j, rt_type::D, kp));
    RIDGE_FT::d_ridgelet_ft(matp, ft_xp, ft_yp, j, rho_x, rho_y);

    // k = -2^(j-1)
    int km = -1 * std::pow(2, j - 1);
    sparse_matrix_t ft_xm, ft_ym;
    RIDGE_FT::x_ridgelet_ft(ft_xm, j, km, rho_x, rho_y);
    RIDGE_FT::y_ridgelet_ft(ft_ym, j, km, rho_x, rho_y);
    auto &matm = insert_sparse(lambda_t(j, rt_type::D, km));
    RIDGE_FT::d_ridgelet_ft(matm, ft_xm, ft_ym, j, rho_x, rho_y);
  }

  // set resolution of physical grid
  int J_min = std::min(Jx, Jy);
  int J_max = std::max(Jx, Jy);
  if (J_max - J_min <= 1) {
    Ny_ = rho_y * (2 << (J_max + 1));
    Nx_ = rho_x * (2 << (J_max + 1));
  } else {
    if (Jx > Jy) {
      Ny_ = 8 * rho_y * (2 << Jy);
      Nx_ = rho_x * (2 << (Jx + 1));
    } else {
      Ny_ = rho_y * (2 << (Jy + 1));
      Nx_ = 8 * rho_x * (2 << (Jx + 1));
    }
  }
}

// --------------------------------------------------------------------------------
inline const std::vector<lambda_t> &
RidgeletFrame::lambdas() const
{
  assert(lambdas_ptr_.use_count() > 0);
  return (*lambdas_ptr_);
}
// --------------------------------------------------------------------------------
inline unsigned int
RidgeletFrame::size() const
{
  return lambdas_ptr_->size();
}
// --------------------------------------------------------------------------------
inline unsigned int
RidgeletFrame::rho_x() const
{
  return rho_x_;
}
// --------------------------------------------------------------------------------
inline unsigned int
RidgeletFrame::rho_y() const
{
  return rho_y_;
}
// --------------------------------------------------------------------------------
inline unsigned int
RidgeletFrame::Jx() const
{
  return Jx_;
}
// --------------------------------------------------------------------------------
inline unsigned int
RidgeletFrame::Jy() const
{
  return Jy_;
}
// --------------------------------------------------------------------------------
inline unsigned int
RidgeletFrame::Nx() const
{
  return Nx_;
}
// --------------------------------------------------------------------------------
inline unsigned int
RidgeletFrame::Ny() const
{
  return Ny_;
}
// --------------------------------------------------------------------------------
inline bool
RidgeletFrame::is_element(const lambda_t &lambda) const
{
  return ((sparse_coeffs_ptr_->find(lambda) != sparse_coeffs_ptr_->end()) ||
          (dense_coeffs_ptr_->find(lambda) != dense_coeffs_ptr_->end()));
}

// --------------------------------------------------------------------------------
inline RidgeletFrame::sparse_matrix_t &
RidgeletFrame::insert_sparse(const lambda_t &lambda)
{
  assert(sparse_coeffs_ptr_.use_count() > 0);
  assert(sparse_coeffs_ptr_->find(lambda) == sparse_coeffs_ptr_->end());
  lambdas_ptr_->push_back(lambda);
  return (*sparse_coeffs_ptr_)[lambda];
}

// --------------------------------------------------------------------------------
inline RidgeletFrame::matrix_t &
RidgeletFrame::insert_dense(const lambda_t &lambda)
{
  assert(dense_coeffs_ptr_.use_count() > 0);
  assert(dense_coeffs_ptr_->find(lambda) == dense_coeffs_ptr_->end());
  // lambdas_ptr_->push_back(lambda);

  return (*dense_coeffs_ptr_)[lambda];
}

// --------------------------------------------------------------------------------
inline const RidgeletFrame::matrix_t &
RidgeletFrame::get_dense(const lambda_t &lambda) const
{
  assert(dense_coeffs_ptr_.use_count() > 0);
  auto it = dense_coeffs_ptr_->find(lambda);
  assert(it != dense_coeffs_ptr_->end());
  return it->second;
}

// --------------------------------------------------------------------------------
inline const RidgeletFrame::sparse_matrix_t &
RidgeletFrame::get_sparse(const lambda_t &lambda) const
{
  assert(sparse_coeffs_ptr_.use_count() > 0);
  auto it = sparse_coeffs_ptr_->find(lambda);
  assert(it != sparse_coeffs_ptr_->end());
  return it->second;
}

// --------------------------------------------------------------------------------
inline double
RidgeletFrame::rf_norm(const lambda_t &lambda) const
{
  if (lambda.t == rt_type::S)
    return std::sqrt(this->get_dense(lambda).cwiseAbs2().sum());
  else
    return std::sqrt(this->get_sparse(lambda).cwiseAbs2().sum());
}
