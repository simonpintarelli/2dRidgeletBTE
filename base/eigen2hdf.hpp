// The MIT License (MIT)
//
// Copyright (c) 2013 James R. Garrison
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
// Modified 2016 Simon Pintarelli

#include <hdf5.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <boost/assert.hpp>
#include <iostream>
#include <stdexcept>

#ifndef __EIGEN3_HDF5__
#define __EIGEN3_HDF5__

namespace eigen2hdf {

template <typename T>
struct DatatypeSpecialization;

// floating-point types

template <>
struct DatatypeSpecialization<float>
{
  static inline hid_t get(void) { return H5T_NATIVE_FLOAT; }
};

template <>
struct DatatypeSpecialization<double>
{
  static inline hid_t get(void) { return H5T_NATIVE_DOUBLE; }
};

template <>
struct DatatypeSpecialization<long double>
{
  static inline hid_t get(void) { return H5T_NATIVE_LDOUBLE; }
};

template <>
struct DatatypeSpecialization<unsigned int>
{
  static inline hid_t get(void) { return H5T_NATIVE_UINT; }
};

template <>
struct DatatypeSpecialization<int>
{
  static inline hid_t get(void) { return H5T_NATIVE_INT; }
};

template <typename T>
struct DatatypeSpecialization<std::complex<T>>
{
 public:
  static inline hid_t get(void)
  {
    static DatatypeSpecialization<std::complex<T>> singleton;
    return singleton.memtype;
  }

 private:
  DatatypeSpecialization()
  {
    herr_t status;

    memtype = H5Tcreate(H5T_COMPOUND, sizeof(std::complex<T>));

    status = H5Tinsert(memtype, "r", 0, DatatypeSpecialization<T>::get());
    assert(status == 0);
    status = H5Tinsert(memtype, "i", sizeof(T), DatatypeSpecialization<T>::get());
    assert(status == 0);
  }

 private:
  hid_t memtype;
};

template <typename Scalar>
class MyTriplet : public Eigen::Triplet<Scalar>
{
 public:
  MyTriplet(void)
      : Eigen::Triplet<Scalar>()
  {
  }

  MyTriplet(const unsigned int &i, const unsigned int &j, const Scalar &v = Scalar(0))
      : Eigen::Triplet<Scalar>(i, j, v)
  {
  }

  static std::size_t offsetof_row(void) { return offsetof(MyTriplet<Scalar>, m_row); }
  static std::size_t offsetof_col(void) { return offsetof(MyTriplet<Scalar>, m_col); }
  static std::size_t offsetof_value(void) { return offsetof(MyTriplet<Scalar>, m_value); }
};

typedef struct crs_t
{
  int _row;
  int _col;
  double _val;

  inline int row() const { return _row; }
  inline int col() const { return _col; }
  inline double value() const { return _val; }
}; /* Compound type */

template <typename SparseMatrixType>
void
save_sparse(const hid_t &file, const std::string &dset_name, const SparseMatrixType &mat)
{
  const unsigned int nnz = mat.nonZeros();
  // allocate memory
  std::vector<crs_t> buffer;
  buffer.reserve(nnz);

  for (int k = 0; k < mat.outerSize(); ++k) {
    for (typename SparseMatrixType::InnerIterator it(mat, k); it; ++it) {
      if (it.value() != 0) {
        crs_t entry;
        entry._row = it.row();
        entry._col = it.col();
        entry._val = it.value();
        buffer.push_back(entry);
      }
    }
  }

  // hdf does not like to write empty sets
  if (buffer.empty()) {
    crs_t entry;
    // prepare dummy entry
    entry._row = 0;
    entry._col = 0;
    entry._val = 0;
    buffer.push_back(entry);
  }

  hid_t filetype, memtype, strtype, space, dset, attribute_id, dataspace_id;
  /* Handles */
  herr_t status;
  hsize_t dims[1] = {buffer.size()};

  memtype = H5Tcreate(H5T_COMPOUND, sizeof(crs_t));
  status = H5Tinsert(memtype, "r", HOFFSET(crs_t, _row), H5T_NATIVE_INT);
  status = H5Tinsert(memtype, "c", HOFFSET(crs_t, _col), H5T_NATIVE_INT);
  status = H5Tinsert(memtype, "v", HOFFSET(crs_t, _val), H5T_NATIVE_DOUBLE);

  filetype = H5Tcreate(H5T_COMPOUND, 8 + 8 + 8);
  status = H5Tinsert(filetype, "r", 0, H5T_STD_I64LE);
  status = H5Tinsert(filetype, "c", 8, H5T_STD_I64LE);
  status = H5Tinsert(filetype, "v", 8 + 8, H5T_IEEE_F64LE);
  /*
   * Create dataspace.  Setting maximum size to NULL sets the maximum
   * size to be the current size.
   */
  space = H5Screate_simple(1, dims, NULL);

  /*
   * Create the dataset and write the compound data to it.
   */
  dset = H5Dcreate(file, dset_name.c_str(), filetype, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Dwrite(dset, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer.data());

  /* store shape information */
  hsize_t dimsA = 2;
  dataspace_id = H5Screate_simple(1, &dimsA, NULL);
  attribute_id = H5Acreate2(dset, "shape", H5T_STD_I32LE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
  int shape[2];
  shape[0] = mat.rows();
  shape[1] = mat.cols();

  status = H5Awrite(attribute_id, H5T_NATIVE_INT, shape);

  status = H5Aclose(attribute_id);
  status = H5Sclose(dataspace_id);
  /*
   * Close and release resources.
   */
  status = H5Dclose(dset);
  status = H5Sclose(space);
  status = H5Tclose(filetype);
}

template <typename SparseMatrixType>
void
load_sparse(const hid_t &file, const std::string &dset_name, SparseMatrixType &mat)
{
  /*
   * Open file and dataset.
   */
  hid_t filetype, memtype, strtype, space, dset, attribute_id, dataspace_id;
  dset = H5Dopen(file, dset_name.c_str(), H5P_DEFAULT);
  herr_t status;
  crs_t *rdata; /* Read buffer */

  int ndims;
  int shape[2];
  /*
   * Get dataspace and allocate memory for read buffer.
   */

  space = H5Dget_space(dset);
  hsize_t dims[1];
  ndims = H5Sget_simple_extent_dims(space, /* IN */
                                    dims,  /* OUT */
                                    NULL);
  rdata = (crs_t *)malloc(dims[0] * sizeof(crs_t));

  /* specify memtype */
  memtype = H5Tcreate(H5T_COMPOUND, sizeof(crs_t));
  status = H5Tinsert(memtype, "r", HOFFSET(crs_t, _row), H5T_NATIVE_INT);
  status = H5Tinsert(memtype, "c", HOFFSET(crs_t, _col), H5T_NATIVE_INT);
  status = H5Tinsert(memtype, "v", HOFFSET(crs_t, _val), H5T_NATIVE_DOUBLE);

  /*
   * Read the data.
   */
  status = H5Dread(dset, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, rdata);

  /*
   * Read attributes
   */
  hid_t attr;
  attr = H5Aopen_name(dset, "shape");
  status = H5Aread(attr, H5T_NATIVE_INT, shape);
  H5Aclose(attr);

  mat.resize(shape[0], shape[1]);
  // mat.reserve( (unsigned int ) ((1.0 * dims[0]) / shape[0])); // allocate
  // memory
  // /*
  //  * write entries to sparse matrix
  //  */
  // unsigned int i;
  // for (i=0; i<dims[0]; i++) {
  //   mat.insert( rdata[i].row, rdata[i].col) = rdata[i].val;
  // }

  mat.setFromTriplets(rdata, rdata + dims[0]);
  mat.makeCompressed();

  /*
   * Close and release resources.  H5Dvlen_reclaim will automatically
   * traverse the structure and free any vlen data (strings in this
   * case).
   */
  status = H5Dvlen_reclaim(memtype, space, H5P_DEFAULT, rdata);
  free(rdata);
  status = H5Dclose(dset);
  status = H5Sclose(space);
  status = H5Tclose(memtype);
}

template <typename Derived>
void
save(const hid_t &loc_id, std::string dset_name, const Eigen::DenseBase<Derived> &mat)
{
  typedef typename Derived::Scalar Scalar;
  hid_t DTYPE = DatatypeSpecialization<Scalar>::get();
  const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> row_major_mat(mat);

  hsize_t dims[] = {(hsize_t)mat.rows(), (hsize_t)mat.cols()};
  hid_t space, dset;
  herr_t status;
  space = H5Screate_simple(2, dims, NULL);
  BOOST_VERIFY(space > 0);
  dset = H5Dcreate2(loc_id, dset_name.c_str(), DTYPE, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  BOOST_VERIFY(dset > 0);

  BOOST_VERIFY(H5Dwrite(dset, DTYPE, H5S_ALL, H5S_ALL, H5P_DEFAULT, row_major_mat.data()) >= 0);

  H5Sclose(space);
  H5Dclose(dset);
}

namespace local {

template <int T>
struct resize
{
};

template <>
struct resize<0>
{
  template <typename M>
  static void apply(M &m, hsize_t *dims, int ndims)
  {
    assert(ndims == 1);
    m.derived().resize(dims[0]);
  }
};

template <>
struct resize<1>
{
  template <typename M>
  static void apply(M &m, hsize_t *dims, int ndims)
  {
    assert(ndims == 2);
    m.derived().resize(dims[0], dims[1]);
  }
};

}  // local

template <typename Derived>
void
load(const hid_t &loc_id, const std::string &name, Eigen::DenseBase<Derived> &mat)
{
  typedef Eigen::DenseBase<Derived> mat_t;
  typedef typename Derived::Scalar Scalar;
  hid_t DTYPE = DatatypeSpecialization<Scalar>::get();

  hid_t dset, space;
  //  dset = H5Dopen(loc_id, name.c_str(), H5P_DEFAULT);
  dset = H5Dopen2(loc_id, name.c_str(), H5P_DEFAULT);
  BOOST_VERIFY(dset > 0);
  hsize_t dims[2];
  space = H5Dget_space(dset);
  int ndims = H5Sget_simple_extent_dims(space,  // IN
                                        dims,   // OUT
                                        NULL);
  hsize_t rows = dims[0];
  hsize_t cols = 0;
  if (ndims == 1) {
    cols = 1;
  } else if (ndims > 1) {
    cols = dims[1];
  } else {
    throw std::runtime_error("eigen2hdf::load: Error reading dataset");
  }

  // read data from hdf
  std::vector<Scalar> data(rows * cols);
  herr_t status;
  status = H5Dread(dset, DTYPE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());

  // store data into Eigen object
  // see http://eigen.tuxfamily.org/dox/TopicFunctionTakingEigenTypes.html
  local::resize<(mat_t::RowsAtCompileTime == -1 || mat_t::RowsAtCompileTime > 1) &&
                (mat_t::ColsAtCompileTime == -1 || mat_t::ColsAtCompileTime > 1)>::apply(mat,
                                                                                         dims,
                                                                                         ndims);

  if (rows == 1 || cols == 1) {
    Eigen::DenseBase<Derived> &mat_ = const_cast<Eigen::DenseBase<Derived> &>(mat);
    // mat_.derived().resize(data.size());
    std::copy(data.begin(), data.end(), mat_.derived().data());
  } else {
    Eigen::DenseBase<Derived> &mat_ = const_cast<Eigen::DenseBase<Derived> &>(mat);
    // mat_.derived().resize(rows, cols);
    mat_ = Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
        data.data(), rows, cols);
  }

  H5Sclose(space);
  H5Dclose(dset);
}

}  // end namespace eigen2hdf

#endif /* __EIGEN3_HDF5__ */
