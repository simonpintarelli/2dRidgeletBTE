#include <boost/assert.hpp>
#include <boost/lexical_cast.hpp>

#ifdef USE_DEPRECATED_BOOST_NPY
#include <boost/numpy.hpp>
#else
#include <boost/python/numpy.hpp>
#endif
#include <boost/program_options.hpp>
#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/tuple.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>

#include "ridgelet/lambda.hpp"
#include "ridgelet/ridgelet_cell_array.hpp"
#include "ridgelet/ridgelet_frame.hpp"
#include "ridgelet/rt.hpp"

namespace bp = boost::python;

#ifdef USE_DEPRECATED_BOOST_NPY
namespace np = boost::numpy;
#else
namespace np = boost::python::numpy;
#endif

class RidgeletFrameWrapper : public RidgeletFrame
{
 public:
  RidgeletFrameWrapper() { /*  empty  */}
  RidgeletFrameWrapper(unsigned int Jx, unsigned int Jy, unsigned int rho_x, unsigned int rho_y)
      : RidgeletFrame(Jx, Jy, rho_x, rho_y)
  { /* empty */ }

  np::ndarray get_dense(const lambda_t& lam) const
  {
    static_assert(RidgeletFrame::matrix_t::IsRowMajor, "expected rowmajor storage");
    typedef RidgeletFrame::matrix_t::Scalar numeric_t;
    auto& eigen_array = RidgeletFrame::get_dense(lam);
    np::ndarray np_arr = np::from_data(
        eigen_array.data(),                  /* data  */
        np::dtype::get_builtin<numeric_t>(), /* data type */
        boost::python::make_tuple(eigen_array.rows(), eigen_array.cols()),
        boost::python::make_tuple(eigen_array.cols() * sizeof(numeric_t), sizeof(numeric_t)),
        boost::python::object(*this));

    return np_arr;
  }

  bp::tuple get_sparse(const lambda_t& lam) const
  {
    auto& sp_mat = RidgeletFrame::get_sparse(lam);

    BOOST_VERIFY(sp_mat.isCompressed());

    typedef RidgeletFrame::sparse_matrix_t sp_mat_t;
    typedef sp_mat_t::StorageIndex index_t;

    static_assert(sp_mat_t::IsRowMajor, "assume CRS matrix");
    static_assert(std::is_same<sp_mat_t::Scalar, double>::value, "type error");

    const index_t* innerIndexPtr = sp_mat.innerIndexPtr();
    const index_t* outerIndexPtr = sp_mat.outerIndexPtr();
    const double* valuePtr = sp_mat.valuePtr();

    np::ndarray idp = np::from_data(innerIndexPtr,
                                    np::dtype::get_builtin<index_t>(),
                                    bp::make_tuple(sp_mat.nonZeros()),
                                    bp::make_tuple(sizeof(index_t)),
                                    boost::python::object(*this));

    np::ndarray odp = np::from_data(outerIndexPtr,
                                    np::dtype::get_builtin<index_t>(),
                                    bp::make_tuple(sp_mat.rows() + 1),
                                    bp::make_tuple(sizeof(index_t)),
                                    boost::python::object(*this));

    np::ndarray vdp = np::from_data(valuePtr,
                                    np::dtype::get_builtin<double>(),
                                    bp::make_tuple(sp_mat.nonZeros()),
                                    bp::make_tuple(sizeof(double)),
                                    boost::python::object(*this));

    return bp::make_tuple(idp, odp, vdp, sp_mat.rows(), sp_mat.cols());
  }
};

class RidgeletCellArrayWrapper : public RidgeletCellArray<>
{
 public:
  typedef RidgeletCellArray<> base_type;

 public:
  RidgeletCellArrayWrapper(const RidgeletFrame& rf)
      : RidgeletCellArray<>(rf)
  { /*  empty */ }

  RidgeletCellArrayWrapper() {}

  np::ndarray get_coeff(int i)
  {
    static_assert(base_type::value_t::IsRowMajor, "expected row-major storage");

    typedef typename base_type::numeric_t numeric_t;
    auto& eigen_array = this->coeffs()[i];
    np::ndarray np_arr = np::from_data(
        reinterpret_cast<void*>(eigen_array.data()), /* data  */
        np::dtype::get_builtin<numeric_t>(),         /* data type */
        boost::python::make_tuple(eigen_array.rows(), eigen_array.cols()),
        boost::python::make_tuple(eigen_array.cols() * sizeof(numeric_t), sizeof(numeric_t)),
        boost::python::object(*this));

    if (!(np_arr.get_flags() & np::ndarray::bitflag::WRITEABLE)) {
      throw std::runtime_error("not writable");
    }

    return np_arr;
  }
};

class RTWrapper
{
 public:
  RTWrapper(const RidgeletFrameWrapper& rt_frame)
      : rt_(rt_frame)
  {}

  RTWrapper() {}

  RidgeletCellArrayWrapper rt(const np::ndarray& src)
  {
    RidgeletCellArrayWrapper rca(rt_.frame());

    const Py_intptr_t* shape = src.get_shape();
    auto Ny = rt_.frame().Ny();
    auto Nx = rt_.frame().Nx();
    BOOST_VERIFY(shape[0] == Ny && shape[1] == Nx);

    typedef typename rt_t::complex_array_t::Scalar numeric_t;

    const numeric_t* data = reinterpret_cast<const numeric_t*>(src.get_data());
    Eigen::Map<const typename rt_t::complex_array_t> msrc(data, Ny, Nx);
    rt_.rt(rca.coeffs(), msrc);
    return rca;
  }

  np::ndarray irt(const RidgeletCellArrayWrapper& src)
  {
    typedef typename rt_t::complex_array_t::Scalar numeric_t;
    static_assert(rt_t::complex_array_t::IsRowMajor, "incompatible array types");

    typedef np::ndarray nd_array_t;
    auto Ny = rt_.frame().Ny();
    auto Nx = rt_.frame().Nx();
    boost::python::tuple shape = boost::python::make_tuple(Ny, Nx);
    np::dtype dt = np::dtype::get_builtin<numeric_t>();
    nd_array_t result = np::empty(shape, dt);
    Eigen::Map<typename rt_t::complex_array_t> dst(
        reinterpret_cast<numeric_t*>(result.get_data()), Ny, Nx);
    rt_.irt(dst, src.coeffs());
    return result;
  }

 private:
  typedef RT<double> rt_t;
  rt_t rt_;
};

#ifdef PYTHON
BOOST_PYTHON_MODULE(libpyFFRT)
{
  using namespace boost::python;
  np::initialize();
  //  numeric::array::set_module_and_type("numpy", "ndarray");
  class_<lambda_t, boost::noncopyable>("Lambda", init<>())
      .def(init<int, rt_type, int>(args("j", "t", "k")))
      .add_property("j", &lambda_t::j)
      .add_property("t", &lambda_t::t)
      .add_property("k", &lambda_t::k)
      .def("__eq__", &lambda_t::operator==)
      .def("__str__", &lambda_t::toString)
      .def("__lt__", &lambda_t::operator<);

  enum_<rt_type>("rt_type")
      .value("s", rt_type::S)
      .value("x", rt_type::X)
      .value("y", rt_type::Y)
      .value("d", rt_type::D);

  class_<std::vector<lambda_t> >("lambda_vec").def(vector_indexing_suite<std::vector<lambda_t> >());

  class_<RidgeletFrame>("RidgeletFrameBase", init<>())
      .def(init<unsigned int, unsigned int, unsigned int, unsigned int>(
          args("Jx", "Jy", "rho_x", "rho_y")))
      .def("Nx", &RidgeletFrame::Nx, "get number of columns")
      .def("Ny", &RidgeletFrame::Ny, "get number of rows")
      .def("rho_x", &RidgeletFrame::rho_x, "rho_x")
      .def("rho_y", &RidgeletFrame::rho_y, "rho_y")
      .def("lambdas",
           &RidgeletFrame::lambdas,
           "lambda set",
           return_value_policy<copy_const_reference>());

  class_<RidgeletFrameWrapper, bases<RidgeletFrame> >("RidgeletFrame", init<>())
      .def(init<unsigned int, unsigned int, unsigned int, unsigned int>(
          args("Jx", "Jy", "rho_x", "rho_y")))
      .def("Nx", &RidgeletFrameWrapper::Nx, "get number of columns")
      .def("Ny", &RidgeletFrameWrapper::Ny, "get number of rows")
      .def("rho_x", &RidgeletFrameWrapper::rho_x, "rho_x")
      .def("rho_y", &RidgeletFrameWrapper::rho_y, "rho_y")
      .def("lambdas",
           &RidgeletFrameWrapper::lambdas,
           "lambda set",
           return_value_policy<copy_const_reference>())
      .def("rf_norm", &RidgeletFrameWrapper::rf_norm, "2-norm of PSI_lambda")
      .def("get_dense", &RidgeletFrameWrapper::get_dense, "dense rt coeff")
      .def("get_sparse", &RidgeletFrameWrapper::get_sparse, "sparse rt coeff");

  class_<RidgeletCellArrayWrapper>("RidgeletCellArray", init<>())
      .def(init<RidgeletFrame>(args("ridgelet_frame")))
      .def("get_coeff", &RidgeletCellArrayWrapper::get_coeff, "get coefficient")
      .def("rf", &RidgeletCellArrayWrapper::rf, "ridgelet frame", return_internal_reference<>());

  class_<RTWrapper, boost::noncopyable>("RT", init<>())
      .def(init<const RidgeletFrameWrapper&>(args("rt_frame")))
      .def("rt", &RTWrapper::rt, "rt")
      .def("irt", &RTWrapper::irt, "irt");
}
#endif
