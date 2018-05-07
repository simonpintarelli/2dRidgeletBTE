// system includes -----------------------------------------------
#include <Eigen/Geometry>
#include <boost/math/constants/constants.hpp>
#include <boost/program_options.hpp>
#include <iostream>
// own includes --------------------------------------------------
#include <base/eigen2hdf.hpp>
#include <base/init.hpp>
#include <base/timer.hpp>
#include <fft/fft2.hpp>
#include <fft/fft2_r2c.hpp>
#include <ridgelet/ridgelet_cell_array.hpp>
#include <ridgelet/ridgelet_frame.hpp>
#include <ridgelet/rt.hpp>

#include <boundary_conditions/inflow_bc.hpp>
#include <operators/operators.hpp>
#include <solver/cg.hpp>
#include <solver/ridgelet_solver.hpp>

using namespace std;

typedef RT<> RT_t;
typedef RT_t::array_t array_t;
typedef RT_t::complex_array_t complex_array_t;
typedef RT_t::rt_coeff_t rt_coeff_t;
typedef RidgeletCellArray<rt_coeff_t> rca_t;

namespace po = boost::program_options;

const double Lx = 1.2;
const double Ly = 1.2;

class RidgeletInflowTest
{
 public:
  RidgeletInflowTest(po::variables_map vm);

  void init_sigma(array_t& Sigma);
  /// initialize fft(f(t=0, x))
  void init_fh0(complex_array_t& Fh);
  /// prescribed f(x) in absorption layer
  void init_fbc(array_t& fbc, double vx, double vy, double ql, double qt);
  /// apply boundary conditions in absorption layer to fft(f(x))
  void apply_bc(complex_array_t& Fh, const array_t& fbc);
  /**
   *
   * @param vx
   * @param vy
   * @param ql    inflow left
   * @param qt    inflow top
   * @param rho   density
   */
  void run(double vx, double vy, double ql, double qt, double rho, int ntsteps);

  RidgeletSolver<rt_coeff_t>& solver() { return rt_solver; }

  void log_cg(bool flag) { log_hist = flag; }

 private:
  RidgeletFrame rf;
  RT_t rt;
  FFTr2c<PlannerR2C> fft;
  RidgeletSolver<rt_coeff_t> rt_solver;
  // parameters
  po::variables_map vm;
  bool log_hist = false;
};

RidgeletInflowTest::RidgeletInflowTest(po::variables_map vm_)
    : vm(vm_)
{
  unsigned int Jx = vm["Jx"].as<unsigned int>();
  unsigned int Jy = vm["Jy"].as<unsigned int>();
  unsigned int rho_x = vm["rx"].as<unsigned int>();
  unsigned int rho_y = vm["ry"].as<unsigned int>();

  rf = RidgeletFrame(Jx, Jy, rho_x, rho_y);
  rt = RT_t(rf);

  const unsigned int Nx = rf.Nx();  // #cols
  const unsigned int Ny = rf.Ny();  // #rows
  init_fftw(fft, FFTW_MEASURE, rf);
  fft.get_plan().create_and_get_plan(Ny / 2, Nx / 2, PlannerR2C::INV);
  fft.get_plan().create_and_get_plan(Ny / 2, Nx / 2, PlannerR2C::FWD);
}

void
RidgeletInflowTest::init_sigma(array_t& Sigma)
{
  double sigma = vm["sigma"].as<double>();
  unsigned int Nx = rf.Nx();
  unsigned int Ny = rf.Ny();

  Eigen::ArrayXd xi = Eigen::ArrayXd::LinSpaced(Nx + 1, 0, Lx).segment(0, Nx);
  Eigen::ArrayXd yi = Eigen::ArrayXd::LinSpaced(Ny + 1, 0, Ly).segment(0, Ny);

  Sigma.setZero();

  auto sigmaf_const = [sigma](double x, double y) {
    double d = std::max(x, y);
    if (d > 1) {
      return sigma;
    } else {
      return 0.0;
    }
  };
  auto sigmaf_lin = [sigma](double x, double y) {
    double d = std::max(x, y);
    if (d > 1 && d <= 1.1)
      return (d - 1) / 0.1 * sigma;
    else
      return sigma;
  };
  auto sigmaf_smooth = [sigma](double x, double y) {
    double d = std::max(x, y);
    double z = d - 1;
    if (d > 1 && d <= 1.2)
      return 400 * std::pow(z, 2) - 4000 * std::pow(z, 3) + 10000 * std::pow(z, 4);
    else
      return 0.;
  };
  std::function<double(double, double)> sigmaf;
  if (vm["sigmaf"].as<std::string>() == "const")
    sigmaf = sigmaf_const;
  else if (vm["sigmaf"].as<std::string>() == "lin")
    sigmaf = sigmaf_lin;
  else if (vm["sigmaf"].as<std::string>() == "smooth")
    sigmaf = sigmaf_smooth;
  else {
    std::cerr << "Option --sigmaf=" << vm["sigmaf"].as<std::string>() << " not found." << std::endl;
  }

  // initialize sigma
  for (unsigned int i = 0; i < Ny; ++i) {
    for (unsigned int j = 0; j < Nx; ++j) {
      if (yi[j] < 1 and yi[i] < 1) continue;
      Sigma(i, j) = sigmaf(xi[j], yi[i]);
    }
  }
}

void
RidgeletInflowTest::init_fh0(complex_array_t& Fh)
{
  unsigned int Nx = rf.Nx();
  unsigned int Ny = rf.Ny();

  Eigen::ArrayXd xi = Eigen::ArrayXd::LinSpaced(Nx / 2 + 1, 0, Lx).segment(0, Nx / 2);
  Eigen::ArrayXd yi = Eigen::ArrayXd::LinSpaced(Ny / 2 + 1, 0, Ly).segment(0, Ny / 2);

  array_t X = xi.transpose().replicate(Ny / 2, 1);
  array_t Y = yi.replicate(1, Nx / 2);
  array_t F = array_t::Ones(Ny / 2, Nx / 2);
  F = (X < 1 && Y < 1).select(F, array_t::Zero(Ny / 2, Nx / 2));
  complex_array_t fh(Ny / 2, Nx / 2);
  fft.ft(fh, F, false);
  Fh.setZero();
  ftcut(Fh, Ny / 2, Nx / 2) = 4 * fh;
}

void
RidgeletInflowTest::init_fbc(array_t& fbc, double vx, double vy, double ql, double qt)
{
  unsigned int Nx = rf.Nx();
  unsigned int Ny = rf.Ny();

  fbc.resize(Ny / 2, Nx / 2);
  make_inflow_source(fbc, vx, vy, Lx, Ly, ql, qt);

  Eigen::ArrayXd xi = Eigen::ArrayXd::LinSpaced(Nx / 2 + 1, 0, Lx).segment(0, Nx / 2);
  Eigen::ArrayXd yi = Eigen::ArrayXd::LinSpaced(Ny / 2 + 1, 0, Ly).segment(0, Ny / 2);

  array_t X = xi.transpose().replicate(Ny / 2, 1);
  array_t Y = yi.replicate(1, Nx / 2);
  fbc = (X < 1 && Y < 1).select(array_t::Zero(Ny / 2, Nx / 2), fbc);
}

void
RidgeletInflowTest::apply_bc(complex_array_t& Fh, const array_t& fbc)
{
  unsigned int Nx = rf.Nx();
  unsigned int Ny = rf.Ny();

  complex_array_t fh = ftcut(Fh, Ny / 2, Nx / 2);
  array_t F(Ny / 2, Nx / 2);
  fft.ift(F, fh);
  F /= 4;
  F = (fbc > 0).select(fbc, F);
  fft.ft(fh, F, false);
  Fh.setZero();
  ftcut(Fh, Ny / 2, Nx / 2) = 4 * fh;
}

void
RidgeletInflowTest::run(double vx, double vy, double ql, double qt, double rho, int ntsteps)
{
  rca_t x(rf);
  double dt = vm["dt"].as<double>();
  int cg_maxit = vm["cg_maxit"].as<int>();
  double cg_tol = vm["cg_tol"].as<double>();

  unsigned int Nx = rf.Nx();
  unsigned int Ny = rf.Ny();

  array_t Sigma(Ny, Nx);
  this->init_sigma(Sigma);

  array_t fbc(Ny / 2, Nx / 2);
  this->init_fbc(fbc, vx, vy, ql, qt);

  hid_t file, group_id;
  if (vm.count("save"))
    file = H5Fcreate("advection_inflow.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  array_t q(Ny, Nx);
  double r = sqrt(vx * vx + vy * vy);
  q.setZero();
  make_inflow_source(q, vx, vy, Lx, Ly, ql, qt);
  q *= (Sigma * r);

  if (vm.count("save")) {
    eigen2hdf::save(file, "q", q);
    eigen2hdf::save(file, "Sigma", Sigma);
    eigen2hdf::save(file, "fbc", fbc);
  }
  complex_array_t qh(Ny, Nx);
  fft.ft(qh, q, false);

  // make F decay smoothly in the absorption layer
  // F *= (array_t::Ones(Ny, Nx) -Sigma/sigma);
  complex_array_t Fh(Ny, Nx);
  this->init_fh0(Fh);

  auto& x_coeffs = x.coeffs();
  rt.rt(x_coeffs, Fh);

  // initialize operators
  // AhA = T'T
  AhAOpsigma AhA(Sigma, vx, vy, Lx, Ly, Nx, Ny, dt);
  // preconditioned operator
  RDTSCTimer timer;
  complex_array_t Bh(Ny, Nx);
  complex_array_t Rhsh(Ny, Nx);  // right hand side
  rca_t b(rf);
  rt_solver = RidgeletSolver<rt_coeff_t>(rf, vx, vy);
  PTransportOpBC<RT_t> A(rt, AhA, vx, vy);
  TransportOperatorBC T(Sigma, vx, vy, Lx, Ly, Nx, Ny, dt);  // required for rhs

  if (vm.count("save")) {
    complex_array_t solh(Ny / 2, Nx / 2);
    solh = ftcut(Fh, Ny / 2, Nx / 2);
    array_t sol(Ny / 2, Nx / 2);
    // scale fourier coefficients by grid_out factor
    // ftcut(solh, Ny, Nx) = Fh;
    fft.ift(sol, solh);
    sol /= 4;
    group_id = H5Gcreate(
        file, boost::lexical_cast<string>(0).c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    eigen2hdf::save(group_id, "sol", sol);
    H5Gclose(group_id);
  }
  // do not print log to stdout
  Logger::GetInstance().detach_stdout();
  for (int i = 1; i <= ntsteps; ++i) {
    this->apply_bc(Fh, fbc);
    // T'*Fh
    Rhsh = Fh + dt * qh;  // rhs in fourier space
    T.apply(Bh, Rhsh, true /* hermitian transpose */);
    rt.rt(b.coeffs(), Bh);

    timer.start();
    Logger::GetInstance().push_prefix(boost::lexical_cast<std::string>(i) + " CGSOLVER");
    rt_solver.solve(x, A, b, cg_tol, cg_maxit, log_hist);
    Logger::GetInstance().pop_prefix();
    auto nc_solve = timer.stop();

    cout << "RidgeletSolver took: " << nc_solve / 1e9 << " Gcycles\n";
    // write arrays to hdf5
    // complex_array_t tmp(Ny, Nx);
    rt.irt(Fh, x.coeffs());

    // eigen output
    if (vm.count("save")) {
      complex_array_t solh(Ny / 2, Nx / 2);
      solh = ftcut(Fh, Ny / 2, Nx / 2);
      array_t sol(Ny / 2, Nx / 2);
      // scale fourier coefficients by grid_out factor
      // ftcut(solh, Ny, Nx) = Fh;
      fft.ift(sol, solh);
      sol /= 4;
      group_id = H5Gcreate(
          file, boost::lexical_cast<string>(i).c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      eigen2hdf::save(group_id, "sol", sol);
      H5Gclose(group_id);
    }
  }
  if (vm.count("save")) H5Fclose(file);
}
