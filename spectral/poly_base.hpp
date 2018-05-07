#pragma once

#include <Eigen/Dense>
#include <boost/multi_array.hpp>
#include <vector>


/**
 *
 * @brief Helper to store (cache) \f$ b_i(x_j) \$
 * where \f$ b_i \f$ is supposed to be a polynomial of degree i.
 *
 */
template <typename NUMERIC>
class PolyBase
{
 public:
  typedef NUMERIC numeric_t;
  typedef Eigen::Array<numeric_t, Eigen::Dynamic, 1> array_t;
  typedef Eigen::Map<const array_t> marray_t;

 public:
  PolyBase(unsigned int n);

  /**
   * @brief
   *
   * @param c coefficients
   * @param y
   */
  void evaluate(const std::vector<numeric_t> &c, std::vector<numeric_t> &y);

  /**
   * @brief returns max_degree + 1
   *
   */
  unsigned int npoly() const;

  /**
   * @brief returns a pointer to the begin of b_i
   *
   * @param n  (degree)
   *
   * @return numeric_t*
   */
  const numeric_t *get(unsigned int n) const;

  /**
   * @brief returns an Eigen::Array of b_i(x),
   *
   *
   * @param n
   *
   * @return Array of length m = size(x)
   */
  marray_t get_array(unsigned int n) const;

  unsigned int get_degree() const { return n_; }
  unsigned int get_npoints() const { return Y_.shape()[1]; }

 protected:
  /// max degree
  unsigned int n_;
  /// b_i(x_j) (i: row index, j: column index)
  boost::multi_array<NUMERIC, 2> Y_;
  bool is_initialized_;
};

// ----------------------------------------------------------------------
template <typename NUMERIC>
PolyBase<NUMERIC>::PolyBase(unsigned int n)
    : n_(n)
    , is_initialized_(false)
{
  /* empty */
}

// ----------------------------------------------------------------------
template <typename NUMERIC>
void
PolyBase<NUMERIC>::evaluate(const std::vector<numeric_t> &c, std::vector<numeric_t> &y)
{
  const unsigned int npts = Y_.shape()[1];
  const unsigned int L = Y_.shape()[0];

  assert(L == c.size());
  assert(npts == y.size());
  assert(is_initialized_);

  for (unsigned int xi = 0; xi < npts; ++xi) {
    y[xi] = 0;
  }

  for (unsigned int l = 0; l < L; ++l) {
    for (unsigned int xi = 0; xi < npts; ++xi) {
      y[xi] += Y_[l][xi] * c[l];
    }
  }
}

// ----------------------------------------------------------------------
template <typename NUMERIC>
const NUMERIC *
PolyBase<NUMERIC>::get(unsigned int n) const
{
  assert(n < Y_.shape()[0]);
  return Y_[n].origin();
}

template <typename NUMERIC>
typename PolyBase<NUMERIC>::marray_t
PolyBase<NUMERIC>::get_array(unsigned int n) const
{
  assert(n < Y_.shape()[0]);
  return marray_t(Y_[n].origin(), Y_.shape()[1]);
}

// ----------------------------------------------------------------------
template <typename NUMERIC>
unsigned int
PolyBase<NUMERIC>::npoly() const
{
  return Y_.shape()[0];
}
