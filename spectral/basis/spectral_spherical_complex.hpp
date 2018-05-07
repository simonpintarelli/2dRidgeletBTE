#pragma once

// ----------------------------------------------------------------------
struct xirc_id_t
{
  typedef xirc_id_t id_t;
  xirc_id_t()
      : l(-1)
  {
  }
  xirc_id_t(int l_)
      : l(l_)
  {
  }

  // copy constructor
  xirc_id_t(const xirc_id_t &other)
      : l(other.l)
  {
  }

  bool operator<(const id_t &other) const { return l < other.l; }

  bool operator==(const id_t &other) const { return l == other.l; }

  friend std::ostream &operator<<(std::ostream &stream, const id_t &x)
  {
    stream << x.to_string();  //(x.t == 0 ? "cos_" : "sin_") << x.l << " ";
    return stream;
  }

  std::string to_string() const { return "exp_ii_" + boost::lexical_cast<std::string>(l); }

  int l;
};

// --------------------------------------------------------------------------------
class XiRC : public weighted<XiRC, false>, public local_::index_policy<xirc_id_t>

{
 public:
  typedef std::complex<double> numeric_t;

 public:
  XiRC(int l)
      : id_(l)
  {
  }
  XiRC()
      : id_(-1)
  {
  }
  numeric_t evaluate(double phi) const
  {
    constexpr const numeric_t ii(0, 1);
    return std::exp(ii * double(id_.l) * phi);
  }

  const id_t &get_id() const { return id_; }

 private:
  id_t id_;
};
