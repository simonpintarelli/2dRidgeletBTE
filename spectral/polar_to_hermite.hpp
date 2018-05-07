#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <array>
#include <boost/assert.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/multi_array.hpp>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

#include "base/array_buffer.hpp"
#include "base/exceptions.hpp"
#include "base/timer.hpp"
#include "filtered_range.hpp"
#ifndef NOHDF5
#include "base/eigen2hdf.hpp"
#endif
#include "quadrature/qhermitew.hpp"
#include "spectral/basis/toolbox/spectral_basis.hpp"
#include "spectral/laguerren_ks.hpp"
#include "spectral/eval_handlers.hpp"
#include "spectral/pl_radial_eval.hpp"


namespace boltzmann {
namespace detail {

/**
 * @brief sort *polar* elements by `k`
 */
struct CMP
{
  template <typename E>
  bool operator()(const E &e1, const E &e2) const
  {
    // typedef typename boost::mpl::at_c<typename E::types_t,0>::type fa_type;
    typedef typename boost::mpl::at_c<typename E::types_t, 1>::type fr_type;

    typename E::Acc::template get<fr_type> fr_accessor;
    // typename E::Acc::template get<fa_type> fa_accessor;

    const auto &idR1 = fr_accessor(e1).get_id();
    const auto &idR2 = fr_accessor(e2).get_id();

    return std::tie(idR1.k, idR1.j) < std::tie(idR2.k, idR2.j);
  }
};

}  // end namespace detail

template <typename PolarBasis, typename HermiteBasis>
class Polar2Hermite
{
 public:
  Polar2Hermite(const PolarBasis &polar_basis, const HermiteBasis &hermite_basis);
#ifndef NOHDF5
  void exportmat(const std::string &fname) const;
#endif

  void to_hermite(std::vector<double> &dst, const std::vector<double> &src) const;

  template <typename DERIVED1, typename DERIVED2>
  void to_hermite(Eigen::DenseBase<DERIVED1> &dst, const Eigen::DenseBase<DERIVED2> &src) const;

  template <typename DERIVED1, typename DERIVED2>
  void to_hermite_T(Eigen::DenseBase<DERIVED1> &dst, const Eigen::DenseBase<DERIVED2> &src) const;

  void to_polar(std::vector<double> &dst, const std::vector<double> &src) const;

  template <typename DERIVED1, typename DERIVED2>
  void to_polar(Eigen::DenseBase<DERIVED1> &dst, const Eigen::DenseBase<DERIVED2> &src) const;

  void to_polar(double *dst, const double *src) const;

  const Eigen::MatrixXd &get_mat(int k) const;

 private:
  /// Transformation matrices for each polynomial degree k
  std::vector<Eigen::MatrixXd> tmatrices_;
  std::vector<unsigned int> offsets_;
  /// Permutation matrix for polar degrees of freedom
  Eigen::SparseMatrix<double> P_;

  Eigen::VectorXd polar_mass_m_;
  thread_local static ::ArrayBuffer<> buf_;
};

template <typename PolarBasis, typename HermiteBasis>
thread_local ::ArrayBuffer<> Polar2Hermite<PolarBasis, HermiteBasis>::buf_;


template <typename PolarBasis, typename HermiteBasis>
Polar2Hermite<PolarBasis, HermiteBasis>::Polar2Hermite(const PolarBasis &polar_basis,
                                                       const HermiteBasis &hermite_basis)
{
  BOOST_VERIFY(polar_basis.n_dofs() == hermite_basis.n_dofs());
  int K = spectral::get_max_k(polar_basis);
  const unsigned int qpts1d = std::max(40, 2 * (K + 1));
  // initialize quadrature
  QHermiteW quad(1.0, qpts1d);

  // typedef std::array<double, 2> point_t;
  typedef boost::multi_array<double, 2> array2d;
  // typedef boost::multi_array<point_t, 2> array2dpts;
  array2d W(boost::extents[qpts1d][qpts1d]);
  // quad. nodes in cartesian coordinates
  array2d X(boost::extents[qpts1d][qpts1d]);
  array2d Y(boost::extents[qpts1d][qpts1d]);
  array2d R(boost::extents[qpts1d][qpts1d]);
  array2d PHI(boost::extents[qpts1d][qpts1d]);

  for (unsigned int i = 0; i < qpts1d; ++i) {
    const double x = quad.pts(i);
    for (unsigned int j = 0; j < qpts1d; ++j) {
      const double y = quad.pts(j);
      X[i][j] = x;
      Y[i][j] = y;
      W[i][j] = quad.wts(i) * quad.wts(j);
      PHI[i][j] = std::atan2(y, x);
      R[i][j] = std::sqrt(y * y + x * x);
    }
  }
  const unsigned int nq = qpts1d * qpts1d;

  //  ...
  PolarBasis basis_k(polar_basis);
  basis_k.sort(detail::CMP());

  tmatrices_.resize(K + 1);
  P_.resize(polar_basis.n_dofs(), polar_basis.n_dofs());

  // evaluate basis functions at quadrature points
  RDTSCTimer timer;
  timer.start();
  LaguerreNKS<double> L(K);
  unsigned int Rlen = R.shape()[0] * R.shape()[1];
  L.compute(R.data(), Rlen, 0.5);

  typedef typename PolarBasis::elem_t polar_elem_t;
  typedef typename boost::mpl::at_c<typename polar_elem_t::types_t, 1>::type radial_elem_t;
  typedef typename boost::mpl::at_c<typename polar_elem_t::types_t, 0>::type ang_elem_t;
  typename polar_elem_t::Acc::template get<radial_elem_t> get_rad;
  typename polar_elem_t::Acc::template get<ang_elem_t> get_ang;

  typedef typename HermiteBasis::elem_t herm_elem_t;
  typedef typename boost::mpl::at_c<typename herm_elem_t::types_t, 0>::type hx_t;
  typedef typename boost::mpl::at_c<typename herm_elem_t::types_t, 1>::type hy_t;

  typename herm_elem_t::Acc::template get<hx_t> get_hx;
  typename herm_elem_t::Acc::template get<hy_t> get_hy;

  // HermiteEvalHandler2d<HermiteNW<double>, herm_elem_t> heval(H);
  auto heval = hermite_evaluator2d<>::make(hermite_basis, quad.pts());
  PLRadialEval<LaguerreNKS<double>, polar_elem_t> leval(L);
  // permutation matrix entries
  std::vector<unsigned int> p_loc(polar_basis.n_dofs());
  // update offset vector
  unsigned int offset = 0;
  offsets_.resize(K + 2);
  // ----------------------------------------------------
  // loop over block-diagonal matrix and compute offsets
  // ----------------------------------------------------
  offsets_[0] = 0;
  for (int k = 0; k <= K; ++k) {
    std::function<bool(const polar_elem_t &)> pred = [&](const polar_elem_t &e) {
      return (get_rad(e).get_id().k == k);
    };
    auto range = filtered_range(basis_k.begin(), basis_k.end(), pred);

    // collect polar elements of polynomial degree k
    std::vector<polar_elem_t> polar_elems;
    for (auto it = std::get<0>(range); it != std::get<1>(range); ++it) {
      polar_elems.push_back(*it);
    }
    offset += polar_elems.size();
    offsets_[k + 1] = offset;
  }

  // --------------------------------------
  // Compute transformation matrix entries
  // --------------------------------------
  timer.start();
#pragma omp parallel for
  // (* load-balancing will be rather poor *)
  for (int k = 0; k <= K; ++k) {
    std::function<bool(const polar_elem_t &)> pred = [&](const polar_elem_t &e) {
      return (get_rad(e).get_id().k == k);
    };
    auto range = filtered_range(basis_k.begin(), basis_k.end(), pred);

    // collect polar elements of polynomial degree k
    std::vector<polar_elem_t> polar_elems;
    for (auto it = std::get<0>(range); it != std::get<1>(range); ++it) {
      polar_elems.push_back(*it);
    }

    // collect hermite elements of polynomial degree k
    std::vector<herm_elem_t> herm_elems;
    std::function<bool(const herm_elem_t &)> pred2 = [&](const herm_elem_t &e) {
      return (get_hx(e).get_id().k + get_hy(e).get_id().k == k);
    };
    auto range_herm = filtered_range(hermite_basis.begin(), hermite_basis.end(), pred2);
    for (auto it = std::get<0>(range_herm); it != std::get<1>(range_herm); ++it) {
      herm_elems.push_back(*it);
    }

    assert(herm_elems.size() == polar_elems.size());

    // init matrix
    auto &Mk = tmatrices_[k];
    unsigned int nk = herm_elems.size();
    Mk.resize(nk, nk);

    // quadrature nodes and weights
    const double *w = W.origin();
    // const double* x = X.origin();
    // const double* y = Y.origin();
    // const double* r = R.origin();
    const double *phi = PHI.origin();

    for (auto itp = polar_elems.begin(); itp != polar_elems.end(); ++itp) {
      // create permutation matrix
      unsigned int dofx = polar_basis.get_dof_index(itp->get_id());
      unsigned int dofy = basis_k.get_dof_index(itp->get_id());
      p_loc[dofy] = dofx;  // permutation
      //      const unsigned int j = get_rad(*itp).get_id().j;
      const unsigned int k = get_rad(*itp).get_id().k;

      for (auto ith = herm_elems.begin(); ith != herm_elems.end(); ++ith) {
        unsigned int dofy2 = hermite_basis.get_dof_index(ith->get_id());  // index
        // quadrature
        double val = 0;
        for (unsigned int q = 0; q < nq; ++q) {
          // const double rloc = r[q];
          val += (w[q] * leval(*itp, q) * get_ang(*itp).evaluate(phi[q]) *
                  heval(*ith, q / qpts1d, q % qpts1d));
        }
        Mk(dofy2 - offsets_[k], dofy - offsets_[k]) = val;
      }
    }
  }

  for (unsigned int dofy = 0; dofy < p_loc.size(); ++dofy) {
    P_.insert(dofy, p_loc[dofy]) = 1;
  }
  P_.makeCompressed();

  // prepare diagonal entries of mass matrix
  const double PI = boost::math::constants::pi<double>();
  // initialize polar mass m
  const unsigned int N = basis_k.n_dofs();
  polar_mass_m_ = Eigen::VectorXd(N);
  for (unsigned int i = 0; i < N; ++i) {
    const auto &elem = basis_k.get_elem(i);
    if (get_ang(elem).get_id().l == 0)
      polar_mass_m_[i] = PI;
    else
      polar_mass_m_[i] = PI / 2;
  }
}

// ---------------------------------------------------------------------
template <typename PolarBasis, typename HermiteBasis>
void
Polar2Hermite<PolarBasis, HermiteBasis>::to_hermite(std::vector<double> &dst,
                                                    const std::vector<double> &src) const
{
  assert(dst.size() == src.size());
  assert(src.size() == (unsigned int)P_.cols());

  typedef Eigen::Map<Eigen::VectorXd> vec_t;
  typedef Eigen::Map<const Eigen::VectorXd> cvec_t;

  cvec_t vsrc(src.data(), src.size());
  vec_t vdst(dst.data(), dst.size());

  this->to_hermite(vdst, vsrc);
}

// ----------------------------------------------------------------------
template <typename PolarBasis, typename HermiteBasis>
template <typename DERIVED1, typename DERIVED2>
void
Polar2Hermite<PolarBasis, HermiteBasis>::to_hermite(Eigen::DenseBase<DERIVED1> &dst,
                                                    const Eigen::DenseBase<DERIVED2> &src) const
{
  const int K = tmatrices_.size();
  const int N = offsets_[K];

  BOOST_ASSERT(dst.size() == N);
  BOOST_ASSERT(src.size() == N);
  auto vpsrc = buf_.get<Eigen::VectorXd>(N);

  // TODO, this is slow
  vpsrc = P_ * src.derived();

  for (int k = 0; k < tmatrices_.size(); ++k) {
    const unsigned int blocksize = offsets_[k + 1] - offsets_[k];
    dst.segment(offsets_[k], blocksize) = tmatrices_[k] * vpsrc.segment(offsets_[k], blocksize);
  }
}


template <typename PolarBasis, typename HermiteBasis>
template <typename DERIVED1, typename DERIVED2>
void
Polar2Hermite<PolarBasis, HermiteBasis>::to_hermite_T(Eigen::DenseBase<DERIVED1> &dst,
                                                      const Eigen::DenseBase<DERIVED2> &src) const
{
  assert(dst.size() == src.size());
  assert(src.size() == (unsigned int)P_.cols());
  typedef Eigen::Map<Eigen::VectorXd> vec_t;
  typedef Eigen::Map<const Eigen::VectorXd> cvec_t;

  const int K = tmatrices_.size();
  const int N = offsets_[K];

  auto vpsrc = buf_.get<Eigen::VectorXd>(N);

  for (int k = 0; k < tmatrices_.size(); ++k) {
    const unsigned int blocksize = offsets_[k + 1] - offsets_[k];
    vpsrc.segment(offsets_[k], blocksize) =
        tmatrices_[k].transpose() * src.segment(offsets_[k], blocksize);
  }

  dst = P_.transpose() * vpsrc;
}


template <typename PolarBasis, typename HermiteBasis>
void
Polar2Hermite<PolarBasis, HermiteBasis>::to_polar(std::vector<double> &dst,
                                                  const std::vector<double> &src) const
{
  assert(dst.size() == src.size());
  assert(src.size() == (unsigned int)P_.cols());
  typedef Eigen::Map<Eigen::VectorXd> vec_t;
  typedef Eigen::Map<const Eigen::VectorXd> cvec_t;
  vec_t vdst(dst.data(), dst.size());
  cvec_t vsrc(src.data(), src.size());

  this->to_polar(vdst, vsrc);
}


template <typename PolarBasis, typename HermiteBasis>
template <typename DERIVED1, typename DERIVED2>
void
Polar2Hermite<PolarBasis, HermiteBasis>::to_polar(Eigen::DenseBase<DERIVED1> &dst,
                                                  const Eigen::DenseBase<DERIVED2> &src) const
{
  const int K = tmatrices_.size();
  const int N = offsets_[K];

  BOOST_ASSERT(dst.size() == N);
  BOOST_ASSERT(src.size() == N);

  // TODO: use buffer memory instead
  auto vpsrc = buf_.get<Eigen::VectorXd>(N);
  for (int k = 0; k < tmatrices_.size(); ++k) {
    const unsigned int blocksize = offsets_[k + 1] - offsets_[k];
    vpsrc.segment(offsets_[k], blocksize) =
        tmatrices_[k].transpose() * src.segment(offsets_[k], blocksize);
  }
  vpsrc = vpsrc.array() / polar_mass_m_.array();

  dst = P_.transpose() * vpsrc;
}


template <typename PolarBasis, typename HermiteBasis>
const Eigen::MatrixXd &
Polar2Hermite<PolarBasis, HermiteBasis>::get_mat(int k) const
{
  BOOST_ASSERT(k < tmatrices_.size());
  return tmatrices_[k];
}

#ifndef NOHDF5
template <typename PolarBasis, typename HermiteBasis>
void
Polar2Hermite<PolarBasis, HermiteBasis>::exportmat(const std::string &fname) const
{
  hid_t h5f = H5Fcreate(fname.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  eigen2hdf::save_sparse(h5f, "P", P_);
  for (unsigned int k = 0; k < tmatrices_.size(); ++k) {
    eigen2hdf::save(h5f, std::to_string(k), tmatrices_[k]);
  }
  H5Fclose(h5f);
}
#endif

}  // end namespace boltzmann
