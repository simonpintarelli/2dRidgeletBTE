#include <mpi.h>
#include <iostream>
#include "base/my_map.hpp"
#include "export/hdf_writer.hpp"

using namespace std;

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);

  int pid, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  int global_size = 100;
  // DoF map
  Map map(global_size);

  // e.g. attempt to write a vector
  const int dim = 3;
  int buf_size = 5;

  // this represents an array [buf_size][dim][local_size]
  // where local_size is determined by map
  BufferType<dim> data(map, buf_size);

  // fill some data
  auto& boost_array = data.array();
  for (int id_buf = 0; id_buf < buf_size; ++id_buf) {
    for (int id_dim = 0; id_dim < dim; ++id_dim) {
      for (int id_loc = 0; id_loc < map.lsize(); ++id_loc) {
        int k = id_loc + map.begin();
        boost_array[id_buf][id_dim][id_loc] = pid * 10000 + 10 * k + id_dim;
      }
    }
  }
  {
    // make sure PHDFWriter is destructed before MPI_Finalize is called!
    PHDFWriter exporter("test.h5");
    exporter.write(data, "D");
  }
  MPI_Finalize();

  return 0;
}
