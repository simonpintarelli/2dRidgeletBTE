#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <cassert>
#include <iostream>
#include <string>
// own includes
#include "base/eigen2hdf.hpp"
#include "base/numbers.hpp"
#include "collision_tensor_galerkin.hpp"
#include "spectral/macroscopic_quantities.hpp"

using namespace std;

namespace boltzmann {

// --------------------------------------------------------------------------------------
CollisionTensorGalerkin::CollisionTensorGalerkin(const basis_t& basis)
    : N_(basis.n_dofs())
    , slices_(basis.n_dofs())
{
  /* empty */
  MQEval Mq(basis);

  // l2 projection
  Ht_.resize(4, N_);
  Ht_.setZero();
  Ht_.block(0, 0, 1, Mq.cmass().rows()) = Mq.cmass().transpose();
  Ht_.block(1, 0, 1, Mq.cenergy().rows()) = Mq.cenergy().transpose();
  Ht_.block(2, 0, 1, Mq.cux().rows()) = Mq.cux().transpose();
  Ht_.block(3, 0, 1, Mq.cuy().rows()) = Mq.cuy().transpose();

  auto make_mass_vdiag = [](const basis_t& basis) {
    Eigen::VectorXd v(basis.n_dofs());
    for (unsigned int i = 0; i < basis.n_dofs(); ++i) {
      auto& elem = basis.get_elem(i);
      typedef typename basis_t::elem_t elem_t;
      typedef typename std::tuple_element<0, typename elem_t::container_t>::type angular_elem_t;
      typedef typename std::tuple_element<1, typename elem_t::container_t>::type radial_elem_t;
      typename elem_t::Acc::template get<angular_elem_t> get_ang;
      typename elem_t::Acc::template get<radial_elem_t> get_rad;
      assert(get_rad(elem).get_id().fw == 0.5);
      if (get_ang(elem).get_id().l == 0) {
        v[i] = numbers::PI;
      } else {
        v[i] = 2 * numbers::PI;
      }
    }
    return v;
  };

  auto s = make_mass_vdiag(basis);
  vec_t sinv = s.array().inverse();
  Sinv_ = diag_t(sinv);
  HtHinv_ = (Ht_ * Sinv_ * Ht_.transpose()).inverse();
}

// --------------------------------------------------------------------------------------
void CollisionTensorGalerkin::add(ptr_t& slice, unsigned int j)
{
  if (j < N_)
    slices_[j] = slice;
  else {
    slices_.push_back(slice);
    ++N_;
  }
}

// --------------------------------------------------------------------------------------
const CollisionTensorGalerkin::sparse_matrix_t& CollisionTensorGalerkin::get(int j) const
{
  return *(this->slices_[j]);
}

// --------------------------------------------------------------------------------------
void CollisionTensorGalerkin::read_hdf5(const char* fname)
{
  ASSERT_MSG(boost::filesystem::exists(fname), "collision tensor file not found");
  slices_.resize(N_);
  hid_t file = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);

  for (unsigned int i = 0; i < N_; ++i) {
    slices_[i] = ptr_t(new sparse_matrix_t(N_, N_));
    eigen2hdf::load_sparse(file, boost::lexical_cast<string>(i), *slices_[i]);
    // make sure this matrix is stored in compressed format
    slices_[i]->makeCompressed();
  }

  ASSERT_MSG((N_ == static_cast<unsigned int>(slices_[0]->rows())) &&
                 (N_ == static_cast<unsigned int>(slices_[0]->cols())),
             "Basis does not match collision tensor");

  H5Fclose(file);
}
}  // end boltzmann
