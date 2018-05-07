#pragma once

#include <malloc.h>
#include <Eigen/Dense>


template <int ALIGN = 32>
struct ArrayBuffer
{
  /**
   * @param n  buffer size in bytes
   *
   */
  ArrayBuffer(unsigned int n)
      : n_(n)
  {
    data_ = memalign(ALIGN, n);
  }

  ArrayBuffer()
      : ArrayBuffer(128)
  { /*  empty */
  }

  // make sure that on copy, new memory is allocated
  ArrayBuffer(const ArrayBuffer &other)
      : ArrayBuffer(other.n_)
  { /* empty */
  }

  /**
   * @brief
   *
   * @param n
   */
  inline void reserve(int n)
  {
    if (n < n_) return;
    if (data_ != NULL) free(data_);
    n_ = n;
    data_ = memalign(ALIGN, n);
  }

  // inline void clear()
  // {
  //   if (data_ != NULL) {
  //     free(data_);
  //     data_ = NULL;
  //   }
  // }

  template <typename ARRAY>
  Eigen::Map<ARRAY> get(unsigned int nx, unsigned int ny)
  {
    typedef typename ARRAY::Scalar numeric_t;
    unsigned int nreq = nx * ny * sizeof(numeric_t);
    this->reserve(nreq);

    return Eigen::Map<ARRAY>(reinterpret_cast<numeric_t *>(data_), nx, ny);
  }

  template <typename VECTOR>
  Eigen::Map<VECTOR> get(unsigned int nx)
  {
    typedef typename VECTOR::Scalar numeric_t;
    unsigned int nreq = nx * sizeof(numeric_t);
    this->reserve(nreq);
    return Eigen::Map<VECTOR>(reinterpret_cast<numeric_t *>(data_), nx);
  }

  // this class is used sometimes used as threadprivate, thus there is no
  // guarantee that the
  // destructor is called
  ~ArrayBuffer()
  {
    if (data_ != NULL) free(data_);
  }

 private:
  void *data_ = NULL;
  unsigned int n_ = 0;
};
