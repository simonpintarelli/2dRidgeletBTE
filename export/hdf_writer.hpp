#pragma once

#include <hdf5.h>
#include <mpi.h>
#include <cstdio>
#include <string>

#include "base/buffered_output_array.hpp"
#include "base/coeff_array.hpp"


class PHDFWriter
{
 public:
  PHDFWriter(const std::string &fname, MPI_Comm comm = MPI_COMM_WORLD);
  ~PHDFWriter();

  /**
   *  @brief write macroscopic quantities (buffered)
   *
   */
  template <int dim>
  void write(const BufferType<dim> &obj, const std::string &dname);

  /**
   *  @brief write full solution vector
   *
   */
  void write(const CoeffArray<double> &arr, const std::string &dname);

 private:
  hid_t file_id_;
  MPI_Comm comm_;
};

template <int dim>
void
PHDFWriter::write(const BufferType<dim> &obj, const std::string &dname)
{
  // datset rank
  const int rank = 3;
  hid_t dcpl_id = H5Pcreate(H5P_DATASET_CREATE);

  // determine chunk size...
  const auto &map = obj.get_map();

  int lsize = map.lsize();
  int max_lsize = -1;
  MPI_Allreduce(&lsize, &max_lsize, 1, MPI_INT, MPI_MAX, comm_);
  BOOST_ASSERT(max_lsize > 0);

  /* setting up chunked output */
  hsize_t chunk_size[rank];
  auto &array = obj.array();
  chunk_size[0] = array.shape()[0];
  chunk_size[1] = array.shape()[1];
  chunk_size[2] = max_lsize;
  H5Pset_chunk(dcpl_id, rank, chunk_size);

  /* create dataspace */
  hsize_t dimsf[rank];  // global size of the array
  dimsf[0] = array.shape()[0];
  dimsf[1] = array.shape()[1];
  dimsf[2] = map.size();

  hid_t filespace = H5Screate_simple(rank, dimsf, NULL);

  hsize_t offset[rank];
  offset[0] = 0;
  offset[1] = 0;
  offset[2] = map.begin();

  /* create dataset with default properties and close filespace */
  hid_t dset_id = H5Dcreate(
      file_id_, dname.c_str(), H5T_NATIVE_DOUBLE, filespace, H5P_DEFAULT, dcpl_id, H5P_DEFAULT);
  H5Sclose(filespace);

  hsize_t count[rank];
  count[0] = dimsf[0];
  count[1] = dimsf[1];
  count[2] = map.lsize();

  hid_t memspace = H5Screate_simple(rank, count, NULL);

  /* select hyperslab */
  filespace = H5Dget_space(dset_id);
  H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, NULL);

  hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

  H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, memspace, filespace, plist_id, array.data());

  /* clean-up */
  H5Pclose(dcpl_id);
  H5Dclose(dset_id);
  H5Sclose(filespace);
  H5Sclose(memspace);
  H5Pclose(plist_id);
}
