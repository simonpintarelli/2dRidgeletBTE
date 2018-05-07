#pragma once

#include <Eigen/Dense>
#include <type_traits>
#include <vector>

#ifdef USE_MPI
#include <mpi.h>
#endif

#include "base/exceptions.hpp"
#include "my_map.hpp"


namespace _local {
template <typename T>
struct TransposeAdaptor
{
  TransposeAdaptor(const T &t_)
      : t(t_){/* empty */};
  const T &operator()() const { return t; }
  const T &t;
};
}  // _local

/*
 *  A collection of vectors, each of length `length`.
 */
template <typename NUMERIC_T = double>
class CoeffArray
{
 private:
  typedef CoeffArray<NUMERIC_T> this_type;

 public:
  typedef NUMERIC_T numeric_t;
  typedef Eigen::Array<NUMERIC_T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> container_t;
  typedef Eigen::Array<NUMERIC_T, Eigen::Dynamic, 1, Eigen::ColMajor> vec_storage_t;
  typedef Eigen::Map<vec_storage_t> view_t;
  typedef Eigen::Map<const vec_storage_t> cview_t;
  typedef Map map_t;

 public:
  CoeffArray(const Map &map, unsigned int length)
      : map_(map)
      , length_(length)
      , storage_(length, map.lsize())
  { /* empty */
  }

  // @{
  /// get data asssigned to index k (local index!)
  view_t get(int k);
  cview_t get(int k) const;
  // @}

  /// the number of coefficients per entry
  unsigned int length() const;

  CoeffArray<NUMERIC_T> &operator=(const _local::TransposeAdaptor<this_type> &wrapped_other);

  const Map &get_map() const;

  container_t &array();
  const container_t &array() const;

  numeric_t *data();
  const numeric_t *data() const;

  _local::TransposeAdaptor<this_type> transpose() const;

  template <typename DERIVED>
  void import_from_full_array(const Eigen::DenseBase<DERIVED> &src);

 protected:
  const Map &map_;
  const unsigned int length_;
  container_t storage_;
};

template <typename NUMERIC_T>
unsigned int
CoeffArray<NUMERIC_T>::length() const
{
  return length_;
}

template <typename NUMERIC_T>
typename CoeffArray<NUMERIC_T>::view_t
CoeffArray<NUMERIC_T>::get(int k)
{
  return view_t(storage_.col(k).data(), length_);
}

template <typename NUMERIC_T>
typename CoeffArray<NUMERIC_T>::cview_t
CoeffArray<NUMERIC_T>::get(int k) const
{
  return cview_t(storage_.col(k).data(), length_);
}

template <typename NUMERIC_T>
typename CoeffArray<NUMERIC_T>::container_t &
CoeffArray<NUMERIC_T>::array()
{
  return storage_;
}

template <typename NUMERIC_T>
const typename CoeffArray<NUMERIC_T>::container_t &
CoeffArray<NUMERIC_T>::array() const
{
  return storage_;
}

template <typename NUMERIC_T>
_local::TransposeAdaptor<typename CoeffArray<NUMERIC_T>::this_type>
CoeffArray<NUMERIC_T>::transpose() const
{
  return _local::TransposeAdaptor<this_type>(*this);
}

template <typename NUMERIC_T>
NUMERIC_T *
CoeffArray<NUMERIC_T>::data()
{
  return storage_.data();
}

template <typename NUMERIC_T>
const NUMERIC_T *
CoeffArray<NUMERIC_T>::data() const
{
  return storage_.data();
}

template <typename NUMERIC_T>
const Map &
CoeffArray<NUMERIC_T>::get_map() const
{
  return map_;
}

template <typename NUMERIC_T>
CoeffArray<NUMERIC_T> &
CoeffArray<NUMERIC_T>::operator=(const _local::TransposeAdaptor<this_type> &wrapped_other)
{
  const auto &other = wrapped_other();
  ASSERT(map_.size() == other.length_);
  ASSERT(length_ == other.map_.size());

  static_assert(std::is_same<NUMERIC_T, double>::value);

  container_t tmp(other.length_, other.map_.size());
  tmp = other.storage_.transpose();

  int nprocs, pid;
  MPI_Comm_rank(map_.comm(), &pid);
  MPI_Comm_size(map_.comm(), &nprocs);

  // counts and offsets, units are no. of elements
  std::vector<int> sendcounts(nprocs);
  std::vector<int> recvcounts(nprocs);
  std::vector<int> sdispls(nprocs);
  std::vector<int> rdispls(nprocs);

  // assemble sendcounts and sdispls
  int soffset = 0;
  for (int i = 0; i < nprocs; ++i) {
    int sendcount = other.map_.lsize() * this->map_.lsize(i);
    sendcounts[i] = sendcount;
    sdispls[i] = soffset;
    soffset += sendcount * sizeof(numeric_t);
  }

  std::vector<MPI_Datatype> send_type(nprocs, MPI_DOUBLE);
  std::vector<MPI_Datatype> recv_type(nprocs);

  // prepare recv types
  for (int i = 0; i < nprocs; ++i) {
    MPI_Type_vector(
        this->map_.lsize(), other.map_.lsize(i), length_, MPI_DOUBLE, recv_type.data() + i);
    // MPI_Type_vector(this->map_.lsize(), other.map_.lsize(i), length_,
    // MPI_DOUBLE,
    // recv_type.data() + i); // (*)
    MPI_Type_commit(recv_type.data() + i);
  }

  // assemble recvcounts and rdispl
  for (int i = 0; i < nprocs; ++i) {
    int recvcount = this->map_.lsize();
    // recvcounts[i] = recvcount; // (*)
    // it does not work with (*)! Why? (also see previous for-loop)
    recvcounts[i] = 1;
    rdispls[i] = other.map_.get_begin(i) * sizeof(numeric_t);
  }

  // TODO: make wrapper for MPI_Datatype!
  int err = MPI_Alltoallw(tmp.data(),
                          sendcounts.data(),
                          sdispls.data(),
                          send_type.data(),
                          storage_.data(),
                          recvcounts.data(),
                          rdispls.data(),
                          recv_type.data(),
                          this->map_.comm());
  BOOST_ASSERT(err == 0);

  return *this;
}

template <typename NUMERIC_T>
template <typename DERIVED>
void
CoeffArray<NUMERIC_T>::import_from_full_array(const Eigen::DenseBase<DERIVED> &src)
{
  static_assert(std::is_same<typename DERIVED::Scalar, NUMERIC_T>::value);
  BOOST_VERIFY(src.rows() == length_);
  storage_ = src.middleCols(map_.begin(), map_.lsize());
}
