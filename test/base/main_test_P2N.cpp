#include <iostream>

#include "base/eigen2hdf.hpp"
#include "spectral/basis/spectral_basis_factory_ks.hpp"
#include "spectral/polar_to_nodal.hpp"

using namespace std;

typedef boltzmann::SpectralBasisFactoryKS basis_factory_t;

int main(int argc, char *argv[])
{
  if (argc < 2) {
    std::cout << "usage: " << argv[0] << " K "
              << "\n";
    return 1;
  }

  int K = atoi(argv[1]);

  typedef basis_factory_t::basis_type basis_type;

  basis_type basis;
  basis_factory_t::create(basis, K, K, 2, true);

  cout << "size(basis) =  " << basis.n_dofs() << "\n";

  typedef boltzmann::Polar2Nodal<basis_type> p2n_t;
  p2n_t p2n;
  p2n.init(basis, 0.5 /*  1/beta  */);

  Eigen::VectorXd cp(basis.n_dofs());
  for (unsigned int i = 0; i < basis.n_dofs(); ++i) {
    cp(i) = std::exp(-20 * float(i) / basis.n_dofs());
  }
  Eigen::MatrixXd cn(K, K);
  p2n.to_nodal(cn, cp);
  Eigen::VectorXd cp2(basis.n_dofs());
  p2n.to_polar(cp2, cn);

  auto diff = cp2;
  diff.setZero();
  for (unsigned int i = 0; i < basis.n_dofs(); ++i) {
    diff(i) = std::abs(cp2(i) - cp(i));
  }
  cout << " sum(err): " << diff.sum() << endl;

  return 0;
}
