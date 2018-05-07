#include <gtest/gtest.h>
#include <boost/math/constants/constants.hpp>
#include <cmath>

#include "spectral/basis/spectral_basis_factory_ks.hpp"
#include "spectral/macroscopic_quantities.hpp"
#include "spectral/polar_to_nodal.hpp"
#include "spectral/quadrature/qhermitew.hpp"

using namespace boltzmann;

TEST(spectral, mqEval)
{
  typedef SpectralBasisFactoryKS::basis_type basis_t;

  double PI = boost::math::constants::pi<double>();

  basis_t basis;
  int K = 30;
  SpectralBasisFactoryKS::create(basis, K);

  Polar2Nodal<> p2n(basis);

  MQEval mq(basis);

  QHermiteW quad(1.0, K);

  double o = 0.1;
  auto f = [PI, o](double x, double y) {
    double cx = (x - o);
    double cy = (y - o);
    return 1. / 2 / PI * std::exp(-cx * cx / 2 - cy * cy / 2);
  };

  Eigen::MatrixXd Nd(K, K);

  for (int i = 0; i < K; ++i) {
    double xi = quad.pts(i);
    double wi = quad.wts(i);
    for (int j = 0; j < K; ++j) {
      double xj = quad.pts(j);
      double wj = quad.wts(j);
      Nd(i, j) = f(xi, xj) * std::sqrt(wi * wj);
    }
  }

  Eigen::VectorXd v(basis.size());

  p2n.to_polar(v, Nd);

  auto evaluator = mq.evaluator();
  evaluator(v);
  const double tol = 1e-12;

  EXPECT_TRUE(std::abs(evaluator.m - 1.0) < tol) << evaluator.m;
  EXPECT_TRUE(std::abs(evaluator.v[0] - o) < tol) << evaluator.v[0];
  EXPECT_TRUE(std::abs(evaluator.v[1] - o) < tol) << evaluator.v[1];

  // std::cout << v << "\n";
}
