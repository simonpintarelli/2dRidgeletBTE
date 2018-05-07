#pragma once

#include <mpi.h>
#include <cmath>

#include <boost/assert.hpp>


/**
 *  @brief Simple (contiguous) distribution of elements.
 *
 *  Detailed description
 *
 *  @param param
 *  @return return type
 */
class Map
{
 public:
  typedef unsigned int index_t;
  typedef unsigned int size_t;

 public:
  Map()
      : size_(0)
      , comm_(0)
  { /* empty */
  }

  explicit Map(size_t global_size, MPI_Comm COMM = MPI_COMM_WORLD)
      : size_(global_size)
      , comm_(COMM)
  {
    int nprocs, pid;
    MPI_Comm_size(COMM, &nprocs);
    MPI_Comm_rank(COMM, &pid);

    blkv_ = _get_blksize(nprocs);
    begin_ = get_begin(pid);
    end_ = get_end(pid);
    lsize_ = end_ - begin_;
  }

  /// map local index ranging from 0 ... lsize to GID (global index)
  index_t GID(index_t lid) const
  {
    BOOST_ASSERT(begin_ + lid < size_);
    return begin_ + lid;
  }

  /// GID begin on this process
  index_t begin() const { return begin_; }

  /// GID end on this process
  index_t end() const { return end_; }

  size_t lsize() const { return lsize_; }

  /// local size
  size_t lsize(int pid) const { return get_end(pid) - get_begin(pid); }

  /// global size
  size_t size() const { return size_; }

  /// GID begin (on process pid)
  index_t get_begin(int pid) const { return pid * blkv_; }

  /// GID end (on process pid)
  index_t get_end(int pid) const { return std::min(size_, (pid + 1) * blkv_); }

  MPI_Comm comm() const { return comm_; }

  bool operator==(const Map &other) const
  {
    return ((this->size_ == other.size_) && (this->begin_ == other.begin_) &&
            (this->end_ == other.end_) && (this->lsize_ == other.lsize_) &&
            (this->comm_ == other.comm_) && (this->blkv_ == other.blkv_));
  }

 private:
  index_t _get_blksize(int nprocs) { return std::ceil(size_ / double(nprocs)); }

 private:
  /// the global size
  size_t size_;
  index_t begin_;
  index_t end_;
  size_t lsize_;
  MPI_Comm comm_;
  size_t blkv_;
};
