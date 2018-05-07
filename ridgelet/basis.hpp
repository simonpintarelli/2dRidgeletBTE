#pragma once

#include <algorithm>
#include <fstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "base/hash_specializations.hpp"
#include "lambda.hpp"
#include "translation.hpp"


struct Elem : public std::tuple<lambda_t, double, double>
{
  typedef std::tuple<lambda_t, double, double> base_type;

  Elem(lambda_t lambda, double ix, double iy)
      : std::tuple<lambda_t, double, double>(lambda, ix, iy)
  { /* empty */ }

  const lambda_t &get_lambda() const { return std::get<0>(*this); }

  double get_iy() const
  {
    // ordering is (y, x)!
    return std::get<1>(*this);
  }

  double get_ix() const { return std::get<2>(*this); }
};

namespace std {
template <>
class hash<Elem>
{
 private:
  typedef Elem value_t;

 public:
  std::size_t operator()(const value_t &value) const
  {
    std::hash<typename Elem::base_type> hb;
    return hb(value);
  }
};
}  // end namespace std

namespace detail_ {

template <typename RTBASIS>
class RTBasisIterator
{
 public:
  typedef long index_t;
  typedef RTBASIS RTBasis_t;
  typedef RTBasisIterator<RTBASIS> own_type;
  typedef typename RTBasis_t::elem_t value_t;

 public:
  RTBasisIterator(const RTBasis_t &basis, index_t pos)
      : basis_(basis)
      , pos_(pos)
  { /* empty */ }

  const value_t &operator*() const { return basis_.get_elem(pos_); }

  const value_t *operator->() const { return &(basis_.get_elem(pos_)); }

  own_type &operator++()
  {
    pos_++;
    return *this;
  }

  own_type &operator--()
  {
    pos_--;
    return *this;
  }

  bool operator<(const own_type &other) const { return pos_ < other.pos_; }

  bool operator<=(const own_type &other) const { return pos_ <= other.pos_; }

  bool operator>(const own_type &other) const { return pos_ > other.pos_; }

  bool operator>=(const own_type &other) const { return pos_ >= other.pos_; }

  explicit operator bool() const { return (pos_ >= 0) && (pos_ < basis_.size()); }

  index_t operator-(const own_type &other) const { return this->pos_ - other.pos_; }

  index_t idx() const
  {
    assert(bool(*this));
    return pos_;
  }

 private:
  index_t pos_;
  const RTBasis_t &basis_;
};
}  // detail_

template <typename ELEM = Elem>
class RTBasis
{
 public:
  typedef ELEM elem_t;
  typedef typename elem_t::base_type tuple_type;
  typedef unsigned int index_t;
  typedef detail_::RTBasisIterator<RTBasis<elem_t>> iterator_t;

 public:
  void insert(const elem_t &elem);
  void insert(const typename std::tuple_element<0, tuple_type>::type &lambda,
              const typename std::tuple_element<1, tuple_type>::type &iy,
              const typename std::tuple_element<2, tuple_type>::type &ix);

  index_t get_index(const elem_t &elem) const;
  index_t get_index(const typename std::tuple_element<0, tuple_type>::type &lambda,
                    const typename std::tuple_element<1, tuple_type>::type &iy,
                    const typename std::tuple_element<2, tuple_type>::type &ix) const;

  const elem_t &get_elem(index_t index) const;
  const std::vector<elem_t> get_elements() const { return elements_; }

  iterator_t get_beg(const typename std::tuple_element<0, tuple_type>::type &lambda) const;
  iterator_t get_end(const typename std::tuple_element<0, tuple_type>::type &lambda) const;

  /// apply lexical ordering
  void sort();
  unsigned int size() const;
  void write_desc(const std::string &fname = "rt_basis.desc") const;

 private:
  std::vector<elem_t> elements_;
  std::unordered_map<elem_t, index_t> e2i_;
  bool is_sorted_ = false;
};

template <typename ELEM>
void
RTBasis<ELEM>::insert(const elem_t &elem)
{
  is_sorted_ = false;
  assert(e2i_.find(elem) == e2i_.end());
  elements_.push_back(elem);
  e2i_[elem] = elements_.size() - 1;
}

template <typename ELEM>
void
RTBasis<ELEM>::insert(const typename std::tuple_element<0, tuple_type>::type &lambda,
                      const typename std::tuple_element<1, tuple_type>::type &iy,
                      const typename std::tuple_element<2, tuple_type>::type &ix)
{
  this->insert(ELEM(lambda, iy, ix));
}

template <typename ELEM>
const ELEM &
RTBasis<ELEM>::get_elem(index_t index) const
{
  BOOST_ASSERT_MSG(index < elements_.size(), "index out of bounds error");
  return elements_[index];
}

template <typename ELEM>
typename RTBasis<ELEM>::index_t
RTBasis<ELEM>::get_index(const elem_t &elem) const
{
  auto it = e2i_.find(elem);
  assert(it != e2i_.end());
  return it->second;
}

template <typename ELEM>
typename RTBasis<ELEM>::index_t
RTBasis<ELEM>::get_index(const typename std::tuple_element<0, tuple_type>::type &lambda,
                         const typename std::tuple_element<1, tuple_type>::type &iy,
                         const typename std::tuple_element<2, tuple_type>::type &ix) const
{
  return this->get_index(ELEM(lambda, iy, ix));
}

template <typename ELEM>
typename RTBasis<ELEM>::iterator_t
RTBasis<ELEM>::get_beg(const typename std::tuple_element<0, tuple_type>::type &lambda) const
{
  if (!is_sorted_)
    throw std::runtime_error("Cannot create RTBasisIterator<..>, call sort() first!");

  auto it = std::lower_bound(
      elements_.begin(), elements_.end(), elem_t(lambda, -1, -1), [](auto &e1, auto &e2) {
        return e1.get_lambda() < e2.get_lambda();
      });

  return iterator_t(*this, it - elements_.begin());
}

template <typename ELEM>
typename RTBasis<ELEM>::iterator_t
RTBasis<ELEM>::get_end(const typename std::tuple_element<0, tuple_type>::type &lambda) const
{
  if (!is_sorted_)
    throw std::runtime_error("Cannot create RTBasisIterator<..>, call sort() first!");

  auto it = std::upper_bound(
      elements_.begin(), elements_.end(), elem_t(lambda, -1, -1), [](auto &e1, auto &e2) {
        return e1.get_lambda() < e2.get_lambda();
      });

  return iterator_t(*this, it - elements_.begin());
}

template <typename ELEM>
void
RTBasis<ELEM>::sort()
{
  std::sort(elements_.begin(), elements_.end());
  is_sorted_ = true;
}

template <typename ELEM>
unsigned int
RTBasis<ELEM>::size() const
{
  return elements_.size();
}

template <typename ELEM>
void
RTBasis<ELEM>::write_desc(const std::string &fname) const
{
  std::ofstream fout(fname);

  for (unsigned int i = 0; i < elements_.size(); ++i) {
    fout << get_elem(i).get_lambda() << " " << get_elem(i).get_iy() << " " << get_elem(i).get_ix()
         << "\n";
  }
  fout.close();
}
