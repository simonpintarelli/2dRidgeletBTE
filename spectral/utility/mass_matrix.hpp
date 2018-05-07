#pragma once

#include "matrix/assembly/velocity_var_form.hpp"

#include <Eigen/Sparse>


namespace boltzmann {

// ----------------------------------------------------------------------
template <typename TEST_BASIS, typename TRIAL_BASIS>
void
make_mass_matrix(Eigen::SparseMatrix<double>& dst,
                 const TEST_BASIS& test_basis,
                 const TRIAL_BASIS& trial_basis,
                 const double beta = 2)
{
  VelocityVarForm<2> velocity_var_form;
  velocity_var_form.init(test_basis, trial_basis, beta);
  const auto& s0 = velocity_var_form.get_s0();

  for (auto it = s0.begin(); it != s0.end(); ++it) {
    int i = it->row;
    int j = it->col;
    double val = it->val;
    dst.insert(i, j) = val;
  }
}

template <typename TEST_BASIS, typename TRIAL_BASIS>
Eigen::SparseMatrix<double>
make_mass_matrix(const TEST_BASIS& test_basis, const TRIAL_BASIS& trial_basis)
{
  assert(test_basis.n_dofs() == trial_basis.n_dofs());
  int N = test_basis.n_dofs();

  Eigen::SparseMatrix<double> M(N, N);
  VelocityVarForm<2> velocity_var_form;
  velocity_var_form.init(test_basis, trial_basis, 2.0);
  const auto& s0 = velocity_var_form.get_s0();

  for (auto it = s0.begin(); it != s0.end(); ++it) {
    int i = it->row;
    int j = it->col;
    double val = it->val;
    M.insert(i, j) = val;
  }

  return M;
}

template <typename TRIAL_BASIS>
Eigen::VectorXd
make_mass_vdiag(const TRIAL_BASIS& basis)
{
  int N = basis.n_dofs();

  Eigen::VectorXd out(N);
  VelocityVarForm<2> velocity_var_form;
  velocity_var_form.init(basis, basis, 2.0);
  const auto& s0 = velocity_var_form.get_s0();

  for (auto it = s0.begin(); it != s0.end(); ++it) {
    int i = it->row;
    int j = it->col;

    if (i != j) throw std::runtime_error("make_mass_vdiag: not diagonal!");
    double val = it->val;
    out[i] = val;
  }

  return out;
}

}  // end namespace boltzmann
