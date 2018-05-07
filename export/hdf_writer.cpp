#include "hdf_writer.hpp"

PHDFWriter::PHDFWriter(const std::string& fname, MPI_Comm comm)
    : comm_(comm)
{
  hid_t plist_id;
  plist_id = H5Pcreate(H5P_FILE_ACCESS);
  BOOST_ASSERT(plist_id > 0);

  /*
   * Set up file access property list with parallel I/O access
   */
  MPI_Info info = MPI_INFO_NULL;
  H5Pset_fapl_mpio(plist_id, comm, info);

  /*
   * Create a new file collectively and release property list identifier.
   */
  file_id_ = H5Fcreate(fname.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
  BOOST_ASSERT(file_id_ > 0);
  H5Pclose(plist_id);
}

void PHDFWriter::write(const CoeffArray<double>& arr, const std::string& dname)
{
  const int rank = 2;
  hid_t dcpl_id = H5Pcreate(H5P_DATASET_CREATE);

  const auto& map = arr.get_map();

  int lsize = map.lsize();
  int max_lsize = -1;
  MPI_Allreduce(&lsize, &max_lsize, 1, MPI_INT, MPI_MAX, comm_);
  BOOST_ASSERT(max_lsize > 0);

  /* setting up chunked output */
  hsize_t chunk_size[rank];
  auto& array = arr.array();
  chunk_size[0] = arr.length();
  chunk_size[1] = max_lsize;
  H5Pset_chunk(dcpl_id, rank, chunk_size);
  /* create dataspace */
  hsize_t dimsf[rank];  // global size of the array
  dimsf[0] = arr.length();
  dimsf[1] = map.size();

  hid_t filespace = H5Screate_simple(rank, dimsf, NULL);

  hsize_t offset[rank];
  offset[0] = 0;
  offset[1] = map.begin();

  /* create dataset with default properties and close filespace */
  hid_t dset_id = H5Dcreate(
      file_id_, dname.c_str(), H5T_NATIVE_DOUBLE, filespace, H5P_DEFAULT, dcpl_id, H5P_DEFAULT);
  H5Sclose(filespace);

  hsize_t count[rank];
  count[0] = dimsf[0];
  count[1] = map.lsize();

  hid_t memspace = H5Screate_simple(rank, count, NULL);

  /* select hyperslab */
  filespace = H5Dget_space(dset_id);
  H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, NULL);

  hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> tmp;
  tmp = array;
  H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, memspace, filespace, plist_id, tmp.data());

  /* clean-up */
  H5Pclose(dcpl_id);
  H5Dclose(dset_id);
  H5Sclose(filespace);
  H5Sclose(memspace);
  H5Pclose(plist_id);
}

PHDFWriter::~PHDFWriter() { H5Fclose(file_id_); }
