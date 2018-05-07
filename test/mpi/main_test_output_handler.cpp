#include <mpi.h>

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "base/my_map.hpp"
#include "spectral/basis/spectral_basis_factory_ks.hpp"
#include "spectral/basis/toolbox/spectral_basis.hpp"
#include "spectral/polar_to_nodal.hpp"

#include "base/import_coeffs.hpp"
#include "export/output_handler.hpp"

typedef boltzmann::SpectralBasisFactoryKS basis_factory_t;
typedef basis_factory_t::basis_type basis_type;

int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);

  if (argc < 2) {
    std::cerr << "syntax: ./" << argv[0] << "L"
              << "\n";
    std::cerr << "L: number of spatial DoFs\n";
    MPI_Finalize();
    return 1;
  }
  int L = atoi(argv[1]);

  int pid, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  basis_type spectral_basis;

  basis_factory_t::create(spectral_basis, "spectral_basis.desc");

  int K = spectral::get_K(spectral_basis);
  int N = K * K;

  Map rte_map(N);  // rte is distributed in v
  Map bte_map(L);  // bte is distributed in x

  CoeffArray<double> rte_vector(rte_map, L);
  CoeffArray<double> bte_vector(bte_map, N);

  load_coeffs_from_file(rte_vector, "output_handler_init.h5", "coeffs");

  // fill bte_vector
  bte_vector = rte_vector.transpose();

  boltzmann::Polar2Nodal<> p2n(spectral_basis);

  OutputHandler output_handler(spectral_basis, bte_map, p2n, 1, 1, 1);

  output_handler.compute(bte_vector, 0, 0.5);

  // create bte vector

  MPI_Finalize();

  return 0;
}
