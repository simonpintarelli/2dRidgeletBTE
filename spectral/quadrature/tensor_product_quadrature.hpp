#pragma once

#include "quadrature_base.hpp"

#include <boost/multi_array.hpp>
#include <vector>


namespace boltzmann {

template <typename Q1, typename Q2>
class TensorProductQuadrature : public Quadrature<2>
{
 public:
  typedef Quadrature<2> base_type;
  using base_type::dim;
  using base_type::coord_type;

 public:
  TensorProductQuadrature(const Q1& q1, const Q2& q2);

 private:
  using base_type::pts_;
  using base_type::wts_;
};

// ---------------------------------------------------------------------------
template <typename Q1, typename Q2>
TensorProductQuadrature<Q1, Q2>::TensorProductQuadrature(const Q1& q1, const Q2& q2)
    : base_type(q1.size() * q2.size())
{
  typedef boost::multi_array_ref<double, 2> wts_ref_t;
  typedef boost::multi_array_ref<coord_type, 2> pts_ref_t;

  wts_ref_t wts_ref(this->wts_.data(), boost::extents[q1.size()][q2.size()]);
  pts_ref_t pts_ref(this->pts_.data(), boost::extents[q1.size()][q2.size()]);

  for (unsigned int i = 0; i < q1.size(); ++i) {
    for (unsigned int j = 0; j < q2.size(); ++j) {
      wts_ref[i][j] = q1.wts(i) * q2.wts(j);
      pts_ref[i][j][0] = q1.pts(i);
      pts_ref[i][j][1] = q2.pts(j);
    }
  }
}

// ----------------------------------------------------------------------------
/**
 * @brief Build tensor product quadrature for R^2, from
 *        quad. rules in polar coordinates.
 *        Stores quadrature points as complex number
 *
 * @tparam Q1 Quad. rule in angular direction
 * @tparam Q2 Quad. rule in radial direction
 *
 */
template <typename Q1, typename Q2>
class TensorProductQuadratureC : public Quadrature<2>
{
 public:
  typedef Quadrature<2> base_type;
  using base_type::dim;
  using base_type::coord_type;

 public:
  TensorProductQuadratureC(const Q1& q1, const Q2& q2);

  /**
   * @brief returns quad. points as std::complex<double>
   *
   * @param i index
   */
  const std::complex<double>& ptsC(unsigned int i) const { return ptsC_[i]; }

  const std::vector<std::complex<double> >& ptsC() const { return ptsC_; }
  const std::array<unsigned int, 2>& dims() const { return dims_; }

 private:
  using base_type::pts_;
  using base_type::wts_;
  std::vector<std::complex<double> > ptsC_;
  std::array<unsigned int, 2> dims_;
};

// ---------------------------------------------------------------------------
template <typename Q1, typename Q2>
TensorProductQuadratureC<Q1, Q2>::TensorProductQuadratureC(const Q1& q1, const Q2& q2)
    : base_type(q1.size() * q2.size())
    , dims_({q1.size(), q2.size()})
{
  typedef boost::multi_array_ref<double, 2> wts_ref_t;
  typedef boost::multi_array_ref<coord_type, 2> pts_ref_t;

  wts_ref_t wts_ref(this->wts_.data(), boost::extents[q1.size()][q2.size()]);
  pts_ref_t pts_ref(this->pts_.data(), boost::extents[q1.size()][q2.size()]);

  for (unsigned int i = 0; i < q1.size(); ++i) {
    for (unsigned int j = 0; j < q2.size(); ++j) {
      wts_ref[i][j] = q1.wts(i) * q2.wts(j);
      pts_ref[i][j][0] = q1.pts(i);
      pts_ref[i][j][1] = q2.pts(j);
    }
  }

  std::complex<double> ii(0, 1);
  ptsC_.resize(pts_.size());
  for (unsigned int i = 0; i < pts_.size(); ++i) {
    ptsC_[i] = pts_[i][1] * std::exp(ii * pts_[i][0]);
  }
}

}  // end namespace boltzmann
