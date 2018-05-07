#pragma once

#include <cmath>
#include <vector>

#include "ridgelet/construction/translation_grid.hpp"
#include "ridgelet/ridgelet_cell_array.hpp"
#include "ridgelet/ridgelet_frame.hpp"
#include "ridgelet/rt.hpp"


class RCLinearize
{
 public:
  RCLinearize(const RidgeletFrame &rf);

  template <typename T>
  std::vector<typename T::Scalar> linearize(const std::vector<T> &coeffs) const;
  /* template<typename DERIVED> */
  /* std::vector<typename DERIVED::Scalar> linearize(const
   * std::vector<Eigen::DenseBase<DERIVED>>&
   * coeffs) const; */

  /// return pair (lambda, t) for a given index pointing into the linearized
  /// coefficient array
  std::tuple<lambda_t, std::tuple<double, double>> get_lambda_tt(unsigned int index) const;

  unsigned int size() const;

  template <typename NUMERIC>
  double get_threshold(const std::vector<NUMERIC> &rca_lin, double keep) const;

  template <typename NUMERIC>
  double get_threshold(std::vector<NUMERIC> &&rca_lin, double keep) const;

  template <typename S>
  double get_threshold(const RidgeletCellArray<S> &rca, double keep) const;

  template <typename S>
  std::vector<double> get_threshold(const RidgeletCellArray<S> &rca,
                                    const std::vector<double> &keepv) const;

  template <typename NUMERIC>
  std::vector<double> get_threshold(std::vector<NUMERIC> &&rca_lin,
                                    const std::vector<double> &keepv) const;

  template <typename S>
  void threshold(RidgeletCellArray<S> &rca, double tre) const;

 private:
  const RidgeletFrame rf_;
  std::vector<double> rf_norms_;
  std::vector<unsigned int> offsets_;
};

RCLinearize::RCLinearize(const RidgeletFrame &rf)
    : rf_(rf)
    , rf_norms_(rf.size())
    , offsets_(rf.size() + 1)
{
  offsets_[0] = 0;
  auto &lambdas = rf.lambdas();
  for (unsigned int i = 0; i < rf.size(); ++i) {
    rf_norms_[i] = rf.rf_norm(lambdas[i]);
    auto tt = tgrid_dim(lambdas[i], rf);
    offsets_[i + 1] = offsets_[i] + std::get<0>(tt) * std::get<1>(tt);
  }
}

template <typename T>
std::vector<typename T::Scalar>
RCLinearize::linearize(const std::vector<T> &coeffs) const
{
  typedef typename T::Scalar numeric_t;
  std::vector<numeric_t> rclin(this->size());

  unsigned int N = rf_norms_.size();
  BOOST_ASSERT(coeffs.size() == N);
  for (unsigned int i = 0; i < N; ++i) {
    double rfnorm = rf_norms_[i];
    unsigned int lsize = coeffs[i].rows() * coeffs[i].cols();
    BOOST_ASSERT(offsets_[i] < rclin.size());
    BOOST_ASSERT(offsets_[i] + lsize <= rclin.size());
    std::transform(coeffs[i].data(),
                   coeffs[i].data() + lsize,
                   rclin.data() + offsets_[i],
                   [&rfnorm](const numeric_t &c) { return c * rfnorm; });
  }
  return rclin;
}

inline std::tuple<lambda_t, std::tuple<double, double>>
RCLinearize::get_lambda_tt(unsigned int index) const
{
  auto it = std::upper_bound(offsets_.begin(), offsets_.end(), index);
  // we need the previous element
  it--;
  int idx = it - offsets_.begin();
  lambda_t ll = rf_.lambdas()[idx];

  // translation grid vector: std::tuple< ty, tx >
  auto tsize = tgrid_dim(ll, rf_);
  int ty_size = std::get<0>(tsize);
  int tx_size = std::get<1>(tsize);

  // RT storage row or column major??
  static_assert(RT<>::rt_coeff_t::IsRowMajor);

  int toffset = index - offsets_[idx];
  int ty = toffset / tx_size;
  int tx = toffset % tx_size;
  std::tuple<double, double> t = {double(ty) / ty_size, double(tx) / tx_size};

  return std::make_tuple(ll, t);
}

unsigned int
RCLinearize::size() const
{
  return *(offsets_.rbegin());
}

template <typename S>
double
RCLinearize::get_threshold(const RidgeletCellArray<S> &rca, double keep) const
{
  BOOST_ASSERT(keep > 0 && keep <= 1);
  return get_threshold(this->linearize(rca.coeffs()), keep);
}

template <typename NUMERIC>
double
RCLinearize::get_threshold(std::vector<NUMERIC> &&rca_lin, double keep) const
{
  typedef NUMERIC numeric_t;
  std::sort(rca_lin.begin(), rca_lin.end(), [](const numeric_t &a, const numeric_t &b) {
    return std::abs(a) < std::abs(b);
  });

  double tre = std::abs(rca_lin[std::floor((1 - keep) * rca_lin.size())]);
  return tre;
}

template <typename NUMERIC>
double
RCLinearize::get_threshold(const std::vector<NUMERIC> &rca_lin, double keep) const
{
  // need to make a copy
  std::vector<NUMERIC> rclin(rca_lin);

  typedef NUMERIC numeric_t;
  std::sort(rclin.begin(), rclin.end(), [](const numeric_t &a, const numeric_t &b) {
    return std::abs(a) < std::abs(b);
  });

  double tre = std::abs(rclin[std::floor((1 - keep) * rclin.size())]);
  return tre;
}

template <typename S>
std::vector<double>
RCLinearize::get_threshold(const RidgeletCellArray<S> &rca, const std::vector<double> &keepv) const
{
  return this->get_threshold(this->linearize(rca.coeffs()), keepv);
}

template <typename NUMERIC>
std::vector<double>
RCLinearize::get_threshold(std::vector<NUMERIC> &&rca_lin, const std::vector<double> &keepv) const
{
  typedef NUMERIC numeric_t;
  std::sort(rca_lin.begin(), rca_lin.end(), [](const numeric_t &a, const numeric_t &b) {
    return std::abs(a) < std::abs(b);
  });

  std::vector<double> ret;
  for (double keep : keepv) {
    BOOST_ASSERT(keep > 0 && keep <= 1);
    double tre = std::abs(rca_lin[std::floor((1 - keep) * rca_lin.size())]);
    ret.push_back(tre);
  }

  return ret;
}

template <typename S>
void
RCLinearize::threshold(RidgeletCellArray<S> &rca, double tre) const
{
  typedef typename RidgeletCellArray<S>::value_t array_t;

  for (unsigned int i = 0; i < rca.size(); ++i) {
    auto &C = rca[i];
    C = ((C * rf_norms_[i]).cwiseAbs() >= tre).select(C, array_t::Zero(C.rows(), C.cols()));
  }
}
