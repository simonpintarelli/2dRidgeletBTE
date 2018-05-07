#pragma once

// system includes ---------------------------------------------------------
#include <Eigen/Dense>
#include <cmath>

// own includes ------------------------------------------------------------
#include "ridgelet_frame.hpp"


template <typename STORAGE = Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
class RidgeletCellArray
{
 public:
  typedef STORAGE value_t;
  typedef RidgeletCellArray<STORAGE> own_type;
  typedef typename STORAGE::Scalar numeric_t;

 public:
  RidgeletCellArray(const RidgeletFrame &rf);
  RidgeletCellArray() {}

  template <class S>
  void resize(const RidgeletCellArray<S> &other);

  const value_t &operator[](int i) const;
  value_t &operator[](int i);
  const std::vector<value_t> &coeffs() const { return data_; }
  std::vector<value_t> &coeffs() { return data_; }

  template <typename S>
  own_type &operator=(const RidgeletCellArray<S> &other);

  template <typename S>
  own_type &operator+=(const RidgeletCellArray<S> &other);

  template <typename S>
  own_type &operator-=(const RidgeletCellArray<S> &other);

  template <typename S>
  own_type &operator*=(const RidgeletCellArray<S> &other);

  template <typename NUMERIC_T>
  own_type &operator*=(NUMERIC_T f);

  template <typename S, typename NUMERIC_T>
  own_type &sadd(const RidgeletCellArray<S> &other, NUMERIC_T a);
  /**
   * this += a*this + b*other
   */
  template <typename S, typename NUMERIC1_T, typename NUMERIC2_T>
  own_type &sadd(NUMERIC1_T a, const RidgeletCellArray<S> &other, NUMERIC2_T b);

  template <typename S, typename NUMERIC_T>
  own_type &sadd(NUMERIC_T a,
                 const RidgeletCellArray<S> &r1,
                 NUMERIC_T a1,
                 const RidgeletCellArray<S> &r2,
                 NUMERIC_T a2);

  double norm() const;
  numeric_t dot(const RidgeletCellArray<STORAGE> &other) const;
  const RidgeletFrame &rf() const { return rf_; }
  unsigned int size() const { return data_.size(); }

 private:
  RidgeletFrame rf_;
  std::vector<value_t> data_;
  Eigen::VectorXd Tx_;
  Eigen::VectorXd Ty_;
  bool T_initialized_ = false;
};

template <typename STORAGE>
RidgeletCellArray<STORAGE>::RidgeletCellArray(const RidgeletFrame &rf)
    : rf_(rf)
{
  data_.resize(rf.size());

  const auto &lambdas = rf_.lambdas();

  std::function<int(int)> pow2p = [](int j) {
    assert(j >= 0);
    if (j == 0)
      return 1;
    else
      return 2 << (j - 1);
  };

  unsigned int N = lambdas.size();
  Tx_.resize(N);
  Ty_.resize(N);
  unsigned int rho_x = rf_.rho_x();
  unsigned int rho_y = rf_.rho_y();
  for (unsigned int i = 0; i < N; ++i) {
    if (lambdas[i].t == rt_type::S) {
      Tx_[i] = 4 * rho_x;
      Ty_[i] = 4 * rho_y;
    } else if (lambdas[i].t == rt_type::D) {
      Tx_[i] = pow2p(lambdas[i].j + 2) * rho_x;
      Ty_[i] = 8 * rho_y;
    } else if (lambdas[i].t == rt_type::X) {
      Tx_[i] = pow2p(lambdas[i].j + 2) * rho_x;
      Ty_[i] = 8 * rho_y;
    } else if (lambdas[i].t == rt_type::Y) {
      Tx_[i] = 8 * rho_x;
      Ty_[i] = pow2p(lambdas[i].j + 2) * rho_y;
    }
    data_[i].resize(Ty_[i], Tx_[i]);
  }
  T_initialized_ = true;
}

template <typename STORAGE>
template <typename S>
void
RidgeletCellArray<STORAGE>::resize(const RidgeletCellArray<S> &other)
{
  data_.resize(other.data_.size());

  for (unsigned int i = 0; i < other.data_.size(); ++i) {
    data_[i].resize(other.data_[i].rows(), other.data_[i].cols());
  }
}

template <typename STORAGE>
typename RidgeletCellArray<STORAGE>::numeric_t
RidgeletCellArray<STORAGE>::dot(const RidgeletCellArray<STORAGE> &other) const
{
  assert(T_initialized_);
  numeric_t sum = 0;
  for (unsigned int i = 0; i < data_.size(); ++i) {
    sum += Tx_[i] * Ty_[i] * (data_[i] * other.data_[i].conjugate()).sum();
  }

  return sum;
}

template <typename STORAGE>
double
RidgeletCellArray<STORAGE>::norm() const
{
  numeric_t v = this->dot(*this);
  return std::sqrt(std::real(v));
}

template <typename STORAGE>
inline const typename RidgeletCellArray<STORAGE>::value_t &RidgeletCellArray<STORAGE>::operator[](
    int i) const
{
  return data_[i];
}

template <typename STORAGE>
inline typename RidgeletCellArray<STORAGE>::value_t &RidgeletCellArray<STORAGE>::operator[](int i)
{
  return data_[i];
}

template <typename STORAGE>
template <typename S>
inline typename RidgeletCellArray<STORAGE>::own_type &
RidgeletCellArray<STORAGE>::operator=(const RidgeletCellArray<S> &other)
{
  data_ = other.data_;
  rf_ = other.rf_;
  Tx_ = other.Tx_;
  Ty_ = other.Ty_;
  T_initialized_ = true;
  return *this;
}

template <typename STORAGE>
template <typename S>
inline typename RidgeletCellArray<STORAGE>::own_type &
RidgeletCellArray<STORAGE>::operator+=(const RidgeletCellArray<S> &other)
{
  for (unsigned int i = 0; i < data_.size(); ++i) {
    data_[i] += other.data_[i];
  }
  return *this;
}

template <typename STORAGE>
template <typename S>
inline typename RidgeletCellArray<STORAGE>::own_type &
RidgeletCellArray<STORAGE>::operator-=(const RidgeletCellArray<S> &other)
{
  for (unsigned int i = 0; i < data_.size(); ++i) {
    data_[i] -= other.data_[i];
  }
  return *this;
}

template <typename STORAGE>
template <typename S>
inline typename RidgeletCellArray<STORAGE>::own_type &
RidgeletCellArray<STORAGE>::operator*=(const RidgeletCellArray<S> &other)
{
  for (unsigned int i = 0; i < data_.size(); ++i) {
    data_[i] *= other.data_[i];
  }
  return *this;
}

template <typename STORAGE>
template <typename NUMERIC_T>
inline typename RidgeletCellArray<STORAGE>::own_type &
RidgeletCellArray<STORAGE>::operator*=(NUMERIC_T f)
{
  for (unsigned int i = 0; i < data_.size(); ++i) {
    data_[i] *= f;
  }
  return *this;
}

template <typename STORAGE>
template <typename S, typename NUMERIC_T>
inline typename RidgeletCellArray<STORAGE>::own_type &
RidgeletCellArray<STORAGE>::sadd(const RidgeletCellArray<S> &other, NUMERIC_T f)
{
  for (unsigned int i = 0; i < data_.size(); ++i) {
    data_[i] += f * other.data_[i];
  }
  return *this;
}

template <typename STORAGE>
template <typename S, typename NUMERIC1_T, typename NUMERIC2_T>
inline typename RidgeletCellArray<STORAGE>::own_type &
RidgeletCellArray<STORAGE>::sadd(NUMERIC1_T a, const RidgeletCellArray<S> &other, NUMERIC2_T b)
{
  for (unsigned int i = 0; i < data_.size(); ++i) {
    data_[i] *= a;
    data_[i] += b * other.data_[i];
  }
}

template <typename STORAGE>
template <typename S, typename NUMERIC_T>
inline typename RidgeletCellArray<STORAGE>::own_type &
RidgeletCellArray<STORAGE>::sadd(NUMERIC_T a,
                                 const RidgeletCellArray<S> &r1,
                                 NUMERIC_T a1,
                                 const RidgeletCellArray<S> &r2,
                                 NUMERIC_T a2)
{
  for (unsigned int i = 0; i < data_.size(); ++i) {
    data_[i] *= a;
    data_[i] += a1 * r1.data_[i] + a2 * r2.data_[i];
  }
}
