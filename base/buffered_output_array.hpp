#pragma once

#include <Eigen/Dense>
#include <boost/mpl/if.hpp>
#include <boost/multi_array.hpp>

// includes for BufferType
#include <Eigen/Dense>
#include <boost/assert.hpp>
#include <type_traits>
#include "base/my_map.hpp"

// debug includes
#include <cmath>
#include <cstdio>


/**
 *  @brief custom 3-d array (used for HDF buffer)
 *
 *  extent: bufsize x dim x n_dofs
 *
 */
template <int DIM, typename NUMERIC_T = double>
class BufferedOutputArray
{
 public:
  typedef NUMERIC_T numeric_t;
  typedef boost::multi_array<NUMERIC_T, 3> storage_t;
  static const int dim = DIM;

 public:
  BufferedOutputArray(int n_dofs, int bufsize)
      : storage_(boost::extents[bufsize][DIM][n_dofs], boost::c_storage_order())
  {
    std::fill(storage_.data(), storage_.data() + bufsize * DIM * n_dofs, -100000.);
  }

 protected:
  storage_t storage_;
};

/**
 *  @brief 3-d Array used as intermediate for HDF output.
 *
 *  The array is distributed among processors contiguously in the 2nd dimension.
 */
template <int dim>
class BufferType : public BufferedOutputArray<dim, double>
{
 private:
  typedef BufferedOutputArray<dim, double> base_t;

 public:
  using storage_t = typename base_t::storage_t;

  const Map &get_map() const { return map_; }
  const storage_t &array() const { return storage_; }
  storage_t &array() { return storage_; }

 public:
  BufferType(const Map &map, int bufsize)
      : base_t(map.lsize(), bufsize)
      , map_(map)
  { /* empty */
  }

  template <typename DERIVED>
  std::enable_if<DERIVED::RowsAtCompileTime * DERIVED::ColsAtCompileTime == dim, void> fill(
      const Eigen::DenseBase<DERIVED> &src, int buf_id, int xid)
  {
    // convert src to c-storage-ordering
    const int length = DERIVED::RowsAtCompileTime * DERIVED::ColsAtCompileTime;

    BOOST_ASSERT(buf_id < storage_.shape()[0]);
    if (!DERIVED::IsRowMajor) {
      // if src is a vector we don't need to care about row/col-major ordering
      typename boost::mpl::if_c<DERIVED::ColsAtCompileTime == 1,
                                Eigen::VectorXd,
                                Eigen::Matrix<double,
                                              DERIVED::RowsAtCompileTime,
                                              DERIVED::ColsAtCompileTime,
                                              Eigen::RowMajor>>::type adapter;
      // this is certainly not fast, but most likely it is not a bottleneck
      // either
      adapter = src;
      for (int i = 0; i < length; ++i) {
        storage_[buf_id][i][xid] = *(adapter.data() + i);
      }
    } else {
      for (int i = 0; i < length; ++i) {
        storage_[buf_id][i][xid] = *(src.derived().data() + i);
      }
    }
  }

  std::enable_if<dim == 1, void> fill(double src, int buf_id, int xid)
  {
    storage_[buf_id][0][xid] = src;
  }

 private:
  using base_t::storage_;
  Map map_;
};
