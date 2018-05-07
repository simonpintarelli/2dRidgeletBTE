#pragma once

#include <string>

#include "base/coeff_array.hpp"
#include "base/eigen2hdf.hpp"
#include "base/exceptions.hpp"


template <typename NUMERIC_T>
void
load_coeffs_from_file(CoeffArray<NUMERIC_T> &dst,
                      const std::string &fname,
                      const std::string &dset = "coeffs")
{
  auto &map = dst.get_map();
  hid_t h5f = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  ASSERT(h5f > 0);
  typename CoeffArray<NUMERIC_T>::container_t tmp;
  eigen2hdf::load(h5f, dset, tmp);

  ASSERT(tmp.rows() == dst.length());
  ASSERT(tmp.cols() == map.size());

  // copy the relevant columns to dst
  dst.import_from_full_array(tmp);
}
