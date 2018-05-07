// system includes -----------------------------------------------
#include <boost/math/constants/constants.hpp>
#include <boost/program_options.hpp>
#include <iostream>
#include <tuple>
// own includes --------------------------------------------------
#include <base/eigen2hdf.hpp>
#include <base/init.hpp>
#include <fft/fft2.hpp>
#include <ridgelet/ridgelet_cell_array.hpp>
#include <ridgelet/ridgelet_frame.hpp>
#include <ridgelet/rt.hpp>

#include <operators/operators.hpp>
#include <solver/cg.hpp>
#include <solver/ridgelet_solver.hpp>

using namespace std;

typedef RT<> RT_t;
typedef RT_t::array_t array_t;
typedef RT_t::complex_array_t complex_array_t;
typedef RT_t::rt_coeff_t rt_coeff_t;
typedef FFTr2c<PlannerR2COD> fft_t;

const double tol = 1e-9;
const int maxit = 400;
double pi = boost::math::constants::pi<double>();
bool save = false;

void dump_frc(const std::vector<rt_coeff_t>& f_rc, const RidgeletFrame& rt, std::string ffname)
{
  const char* fname = ffname.c_str();
  hid_t file = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  for (unsigned int i = 0; i < f_rc.size(); ++i) {
    stringstream ss;
    ss << rt.lambdas()[i];
    string slam = ss.str();
    eigen2hdf::save(file, slam, f_rc[i]);
  }
  H5Fclose(file);
  cout << "Written f(lambda, t) to " << fname << "\n";
}

// --------------------------------------------------------------------------------
// solve in Ridgelet domain
template <typename ARRAY_T>
std::tuple<int, double> solve_rt(
    ARRAY_T& Fh, const RidgeletFrame& rf, const double dt, Eigen::Vector2d& v, bool log)
{
  const unsigned int Nx = rf.Nx();  // #cols
  const unsigned int Ny = rf.Ny();  // #rows

  RT_t rt(rf);
  typedef RidgeletCellArray<rt_coeff_t> rca_t;
  rca_t rt_cell_array(rf);
  auto& rt_coeffs = rt_cell_array.coeffs();
  rt.rt(rt_coeffs, Fh);

  double vx = v[0];
  double vy = v[1];

  // since vx, vy = 0, this has no effect
  double Lx = 1;
  double Ly = 1;

  // initialize operators
  AhAOp AhA(vx, vy, Lx, Ly, Nx, Ny, dt);
  PTransportOp<RT_t> A(rt, AhA, vx, vy);

  RidgeletSolver<rt_coeff_t> rt_solver(rf, vx, vy, dt);
  // assert(vx == 0);
  // assert(vy == 0);
  // have to apply A.T to rhs otherwise!!!

  /////////////
  // b = A*x //
  /////////////
  rca_t x(rf);
  x.resize(rt_cell_array);
  rca_t b(rf);
  b.resize(rt_cell_array);
  x = rt_cell_array;

  // b = A*x
  // Note if v=[0,0]: A is the identity operator
  b = x;

  /* cout << "RT_SOLVER::tol  : " << tol << "\n"; */
  /* cout << "RT_SOLVER::maxit: " << maxit << "\n"; */
  Logger::GetInstance().push_prefix("RTCG");
  rt_solver.set_log(log);
  rt_solver.solve(x, A, b, tol, maxit);
  Logger::GetInstance().clear_prefix();

  if (save) {
    rt.irt(Fh, x.coeffs());
    ARRAY_T Fh2 = ftcut(Fh, Ny / 2, Nx / 2);
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> X(Ny / 2, Nx / 2);
    fft_t fft;
    fft.ift(X, Fh2);
    hid_t file = H5Fcreate("cg-result.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    eigen2hdf::save(file, "X", X);
    H5Fclose(file);
  }

  return std::make_tuple(rt_solver.iter(), rt_solver.relres());

  /* cout << "RT CG::RELRES " << rt_solver.relres() << "\n" */
  /*      << "RT CG::ITER "   << rt_solver.iter() << "\n"; */
}

// --------------------------------------------------------------------------------
// solve in Ridgelet domain
template <typename ARRAY_T>
std::tuple<int, double> solve_rtnop(
    ARRAY_T& Fh, const RidgeletFrame& rf, const double dt, Eigen::Vector2d& v, bool log)
{
  const unsigned int Nx = rf.Nx();  // #cols
  const unsigned int Ny = rf.Ny();  // #rows

  RT_t rt(rf);
  typedef RidgeletCellArray<rt_coeff_t> rca_t;
  rca_t rt_cell_array(rf);
  auto& rt_coeffs = rt_cell_array.coeffs();
  rt.rt(rt_coeffs, Fh);

  double vx = v[0];
  double vy = v[1];

  // since vx, vy = 0, this has no effect
  double Lx = 1;
  double Ly = 1;

  // initialize operators
  AhAOp AhA(vx, vy, Lx, Ly, Nx, Ny, dt);
  PTransportOpId<RT_t> A(rt, AhA, vx, vy);

  RidgeletSolverNOP<rt_coeff_t> rt_solver(rf, vx, vy, dt);
  // assert(vx == 0);
  // assert(vy == 0);
  // have to apply A.T to rhs otherwise!!!

  /////////////
  // b = A*x //
  /////////////
  rca_t x(rf);
  x.resize(rt_cell_array);
  rca_t b(rf);
  b.resize(rt_cell_array);
  x = rt_cell_array;

  // b = A*x
  // Note if v=[0,0]: A is the identity operator
  b = x;

  /* cout << "RT_SOLVER::tol  : " << tol << "\n"; */
  /* cout << "RT_SOLVER::maxit: " << maxit << "\n"; */
  Logger::GetInstance().push_prefix("RTCGNOP");
  rt_solver.set_log(log);
  rt_solver.solve(x, A, b, tol, maxit);
  Logger::GetInstance().clear_prefix();

  return std::make_tuple(rt_solver.iter(), rt_solver.relres());

  /* cout << "RT CG::RELRES " << rt_solver.relres() << "\n" */
  /*      << "RT CG::ITER "   << rt_solver.iter() << "\n"; */
}

// --------------------------------------------------------------------------------
// solve in Fourier domain
template <typename ARRAY_T>
std::tuple<int, double> solve_ft(ARRAY_T& Fh, const double dt, Eigen::Vector2d& v, bool log)
{
  const unsigned int Nx = Fh.cols();  // #cols
  const unsigned int Ny = Fh.rows();  // #rows

  double vx = v[0];
  double vy = v[1];
  // since vx, vy = 0, this has no effect
  double Lx = 1;
  double Ly = 1;

  // initialize operators
  AhAOp A(vx, vy, Lx, Ly, Nx, Ny, dt);

  /////////////
  // b = A*x //
  /////////////
  ARRAY_T x(Ny, Nx);
  x = Fh;

  ARRAY_T b(Ny, Nx);
  b = x;

  Logger::GetInstance().push_prefix("FTCG");
  CG cg;
  cg.set_log(log);
  cg.solve(x, A, b, tol, maxit);
  Logger::GetInstance().clear_prefix();

  return std::make_tuple(cg.iter(), cg.relres());

  /* cout << "FT CG::RELRES " << cg.relres() << "\n" */
  /*      << "FT CG::ITER "  << cg.iter() << "\n"; */
}

int main(int argc, char* argv[])
{
  SOURCE_INFO();

  namespace po = boost::program_options;

  unsigned int Jx, Jy, rho_x, rho_y;
  int nv;
  double dt, r;
  bool log;

  po::options_description options("options");
  options.add_options()("help", "produce help message")
      ("Jx,i", po::value<unsigned int>(&Jx)->default_value(3), "Jx")
      ("Jy,j", po::value<unsigned int>(&Jy)->default_value(3), "Jy")
      ("rx,x", po::value<unsigned int>(&rho_x)->default_value(1), "rho_x")
      ("ry,y", po::value<unsigned int>(&rho_y)->default_value(1), "rho_x")
      ("rad,r", po::value<double>(&r)->default_value(1), "|v|")
      /* ("vx", po::value<double>(&vx)->default_value(1), "vx") */
      /* ("vy", po::value<double>(&vy)->default_value(0), "vy") */
      ("dt,t", po::value<double>(&dt)->default_value(0.1), "dt")
      ("nv", po::value<int>(&nv)->default_value(6), "#directons in velocity")
      ("log", po::value<bool>(&log)->default_value(false), "log cg history")
      ("smooth", "use smooth initial condition")
      ("save", "save solution");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, options), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << options << "\n";
    return 0;
  }

  if (vm.count("save")) save = true;

  cout << setw(20) << "Jx"
       << ": " << Jx << "\n"
       << setw(20) << "Jy"
       << ": " << Jy << "\n"
       << setw(20) << "rho_x"
       << ": " << rho_x << "\n"
       << setw(20) << "rho_y"
       << ": " << rho_y << "\n"
       << setw(20) << "dt"
       << ": " << dt << "\n"
       << setw(20) << "|v|"
       << ": " << r << "\n";

  RidgeletFrame rf(Jx, Jy, rho_x, rho_y);

  const unsigned int Nx = rf.Nx();  // #cols
  const unsigned int Ny = rf.Ny();  // #rows
  cout << "Nx: " << Nx << "\n";
  cout << "Ny: " << Ny << "\n";

  auto phi = 2 * pi * Eigen::VectorXd::LinSpaced(nv + 1, 0, 1);

  for (int i = 0; i < nv; ++i) {
    cout << "---------- phi = " << phi[i] << " ----------\n";
    const double vx = std::cos(phi[i]) * r;
    const double vy = std::sin(phi[i]) * r;
    Eigen::Vector2d v;
    v[0] = vx;
    v[1] = vy;

    Eigen::ArrayXd xi = Eigen::ArrayXd::LinSpaced(Nx, 0, 1);
    Eigen::ArrayXd yi = Eigen::ArrayXd::LinSpaced(Ny, 0, 1);
    array_t F(Ny, Nx);
    F = (-100 * ((yi - 0.5)).cwiseAbs2()).exp().replicate(1, xi.rows()) *
        (-100 * ((xi - 0.5).cwiseAbs2())).exp().transpose().replicate(yi.rows(), 1);
    if (!vm.count("smooth")) {
      F = (((yi - 0.5).replicate(1, xi.rows()) * vy -
            (xi - 0.5).transpose().replicate(yi.rows(), 1) * vx) <=
           0).select(F, Eigen::ArrayXXd::Zero(Ny, Nx));
    }

    hid_t file = H5Fcreate("cg.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    eigen2hdf::save(file, "x0", F);
    H5Fclose(file);

    // F.setOnes();

    // ----------------------------------------
    /* double n[2] = {1,1}; */
    /* F = ((n[0]*(yi.replicate(1, xi.rows())-0.5*pi) + n[1]*(xi.transpose().replicate(yi.rows(),
     * 1)-0.5*pi)) > 1e-8) */
    /*   .select(F, array_t::Zero(Ny, Nx)); */
    fft_t fft;
    complex_array_t Fh(Ny, Nx);
    fft.ft(Fh, F, false);

    auto ret_rt = solve_rt(Fh, rf, dt, v, log);
    auto ret_ft = solve_ft(Fh, dt, v, log);
    auto ret_rtnop = solve_rtnop(Fh, rf, dt, v, log);

    cout << "RTCG " << std::scientific << std::setprecision(3) << std::setw(7) << vx << " "
         << std::setw(7) << vy << " " << std::setw(7) << std::get<1>(ret_rt) << " " << std::setw(7)
         << std::get<0>(ret_rt) << "\n";
    cout << "RTCGNOP " << std::scientific << std::setprecision(3) << std::setw(7) << vx << " "
         << std::setw(7) << vy << " " << std::setw(7) << std::get<1>(ret_rtnop) << " "
         << std::setw(7) << std::get<0>(ret_rtnop) << "\n";
    cout << "FTCG " << std::scientific << std::setprecision(3) << std::setw(7) << vx << " "
         << std::setw(7) << vy << " " << std::setw(7) << std::get<1>(ret_ft) << " " << std::setw(7)
         << std::get<0>(ret_ft) << "\n";

    /* std::vector<double> diagp = make_inv_diagonal_preconditioner(rf, dt*vx, dt*vy); */
    /* cout  << "\n"; */
    /* for (int i = 0; i < diagp.size(); ++i) { */
    /*   cout << diagp[i] << "\n"; */
    /* } */
    /* cout << "\n"; */
  }

  return 0;
}
