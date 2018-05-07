#include <gtest/gtest.h>
#include <omp.h>
#include <cmath>

#include "spectral/basis/spectral_basis_factory_ks.hpp"
#include "spectral/polar_to_nodal.hpp"

using namespace boltzmann;

TEST(spectral, p2n)
{
  typedef SpectralBasisFactoryKS::basis_type basis_t;

  basis_t basis;
  int K = 20;
  SpectralBasisFactoryKS::create(basis, K);

  Polar2Nodal<> p2n(basis);

  auto p2h = p2n.get_p2h();

  Eigen::VectorXd v(basis.size());
  v.setZero();
  for (int i = 0; i < std::min((long int)1, v.size()); ++i) {
    v[i] = 1.0;
  }

#pragma omp parallel
  {
    // for (int k = 0; k <= K; ++k) {
    //   auto M = p2h->get_mat(k);
    //   int n = M.rows();
    //   Eigen::MatrixXd diff = M.transpose()*M - Eigen::MatrixXd::Identity(n, n);
    //   double kdiff = diff.cwiseAbs().sum();
    //   EXPECT_TRUE(kdiff < 1e-12) << kdiff << k;
    // }

    Eigen::VectorXd vh(basis.size());
    p2h->to_hermite(vh, v);
    Eigen::VectorXd vih(basis.size());
    p2h->to_polar(vih, vh);

    double diffp2h = (v - vih).cwiseAbs().sum();
    EXPECT_TRUE(diffp2h < 1e-12) << diffp2h;
  }

  {
    Eigen::VectorXd vh(basis.size());
    Eigen::VectorXd vih(basis.size());
    vh.setZero();
    for (int i = 0; i < v.size(); ++i) {
      vh[i] = 1;
    }
    Eigen::MatrixXd N(K, K);

    auto h2n = p2n.get_h2n();
    h2n->to_nodal(N, vh);
    h2n->to_hermite(vih, N);
    double diffh2n = (vh - vih).cwiseAbs().sum();
    EXPECT_TRUE(diffh2n < 1e-12) << diffh2n;
  }

  {
    Eigen::VectorXd vi(basis.size());
    Eigen::MatrixXd N(K, K);
    p2n.to_nodal(N, v);
    p2n.to_polar(vi, N);

    double diff = (v - vi).cwiseAbs().sum();
    EXPECT_TRUE(diff < 1e-12) << diff;
  }
}
