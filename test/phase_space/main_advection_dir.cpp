/**
 * @file   main_advection_dir.cpp
 * @author Simon Pintarelli <simon@thinkpadX1>
 * @date   Wed Mar 30 18:30:47 2016
 *
 * @brief  solve advection equation separately for each direction (quad points)
 *
 *
 */

// system includes -----------------------------------------------
#include <boost/math/constants/constants.hpp>
#include <boost/program_options.hpp>
#include <iostream>
// own includes --------------------------------------------------
#include <base/eigen2hdf.hpp>
#include <base/init.hpp>
#include <base/timer.hpp>
#include <fft/fft2.hpp>
#include <fft/fft2_r2c.hpp>
#include <ridgelet/init_fftw.hpp>
#include <ridgelet/ridgelet_cell_array.hpp>
#include <ridgelet/ridgelet_frame.hpp>
#include <ridgelet/rt.hpp>

#include <omp.h>
#include <operators/operators.hpp>
#include <solver/cg.hpp>
#include <solver/ridgelet_solver.hpp>
#include <spectral/quadrature/gauss_hermite_roots.hpp>

using namespace std;
typedef FFTr2c<PlannerR2C> fft_t;
// TODO: check if RT coeffs are real valued in this case
typedef RT<double, RidgeletFrame, fft_t> RT_t;
typedef RT_t::array_t array_t;
typedef RT_t::complex_array_t complex_array_t;
typedef RT_t::rt_coeff_t rt_coeff_t;

int main(int argc, char* argv[])
{
  SOURCE_INFO();

  namespace po = boost::program_options;

  unsigned int Jx, Jy, rho_x, rho_y;
  double dt;
  unsigned int f;
  int K;

  unsigned int cg_maxit;
  double cg_reltol;

  po::options_description options("options");
  options.add_options()("help", "produce help message")
      ("Jx,i", po::value<unsigned int>(&Jx)->default_value(3), "Jx")
      ("Jy,j", po::value<unsigned int>(&Jy)->default_value(3), "Jy")
      ("rx,x", po::value<unsigned int>(&rho_x)->default_value(1), "rho_x")
      ("ry,y", po::value<unsigned int>(&rho_y)->default_value(1), "rho_x")
      ("dt,t", po::value<double>(&dt)->default_value(0.1), "dt")
      ("deg,K", po::value<int>(&K)->default_value(10), "poly. deg.")
      ("f", po::value<unsigned int>(&f)->default_value(2), "grid out factor")
      ("maxiter", po::value<unsigned int>(&cg_maxit)->default_value(40), "cg::maxiter")
      ("reltol", po::value<double>(&cg_reltol)->default_value(1e-4), "cg::reltol");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, options), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << options << "\n";
    return 0;
  }

  cout << "CMD::";
  for (int i = 0; i < argc; ++i) {
    cout << argv[i] << " ";
  }
  cout << "\n";

  cout << setw(20) << "Jx"
       << ": " << Jx << "\n"
       << setw(20) << "Jy"
       << ": " << Jy << "\n"
       << setw(20) << "rho_x"
       << ": " << rho_x << "\n"
       << setw(20) << "rho_y"
       << ": " << rho_y << "\n"
       << setw(20) << "K"
       << ": " << K << "\n"
       << setw(20) << "dt"
       << ": " << dt << "\n";

  RidgeletFrame rf(Jx, Jy, rho_x, rho_y);
  const unsigned int Nx = rf.Nx();  // #cols
  const unsigned int Ny = rf.Ny();  // #rows
  cout << "Nx: " << Nx << "\n";
  cout << "Ny: " << Ny << "\n";

  fft_t fft;
  init_fftw(fft, FFTW_MEASURE, rf);
  fft.get_plan().create_and_get_plan(f * Ny, f * Nx, PlannerR2C::INV);
  fft.get_plan().create_and_get_plan(f * Ny, f * Nx, PlannerR2C::FWD);

  std::vector<double> vi(K);
  boltzmann::gauss_hermite_roots(vi, K);

  Eigen::ArrayXd xi = Eigen::ArrayXd::LinSpaced(Nx + 1, 0, 1).segment(0, Nx);
  Eigen::ArrayXd yi = Eigen::ArrayXd::LinSpaced(Ny + 1, 0, 1).segment(0, Ny);

  array_t F =
      (xi.transpose().replicate(Ny, 1)).binaryExpr(yi.replicate(1, Nx), [](double x, double y) {
        return std::exp(-300 * (std::pow(x - 0.5, 2) + std::pow(y - 0.5, 2)));
      });

  RT_t rt(rf);

  complex_array_t Fh(Ny, Nx);
  fft.ft(Fh, F, false);
  typedef RidgeletCellArray<rt_coeff_t> rca_t;
  rca_t rt_cell_array(rf);
  auto& rt_coeffs = rt_cell_array.coeffs();
  rt.rt(rt_coeffs, Fh);

  hid_t file = H5Fcreate("advection_dir.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  cout << "write results to `advection_dir.h5`"
       << "\n";
  cout << "max thread: " << omp_get_max_threads() << "\n";
  cout << "RT_SOLVER::tol  : " << cg_reltol << "\n";
  cout << "RT_SOLVER::maxit: " << cg_maxit << "\n";

  eigen2hdf::save(file, "F", F);
  eigen2hdf::save(file, "Fh", Fh);
  cout << "dt:" << dt << "\n";
  const double Lx = 1;
  const double Ly = 1;

  Eigen::MatrixXi ITER_MAT(K, K);
  Eigen::MatrixXd RELRES_MAT(K, K);  // rel. residual
  Eigen::MatrixXd TIME_MAT(K, K);    // [GCycle]

#pragma omp parallel for
  for (int qx = 0; qx < K; ++qx) {
    for (int qy = 0; qy < K; ++qy) {
      const double vx = vi[qx];
      const double vy = vi[qy];
      AhAOp AhA(vx, vy, Lx, Ly, Nx, Ny, dt);
      // preconditioned operator
      RDTSCTimer timer;

      PTransportOp<RT_t> A(rt, AhA, vx, vy);
      TransportOperator T(vx, vy, Lx, Ly, Nx, Ny, dt);  // required for rhs
      // T'*Fh
      complex_array_t Bh(Ny, Nx);
      T.apply(Bh, Fh, true /* hermitian transpose */);
      rca_t b(rf);
      rt.rt(b.coeffs(), Bh);
      RidgeletSolver<rt_coeff_t> rt_solver(rf, vx, vy);
      rca_t x(rf);
      x.resize(rt_cell_array);
      x = rt_cell_array;
      timer.start();
      rt_solver.solve(x, A, b, cg_reltol, cg_maxit);
      auto nc_solve = timer.stop();

      RELRES_MAT(qx, qy) = rt_solver.relres();
      ITER_MAT(qx, qy) = rt_solver.iter();
      TIME_MAT(qx, qy) = nc_solve / 1e9;

      // #pragma omp critical
      //       {
      //         cout << "vx: " << vx << "\n";
      //         cout << "vy: " << vy << "\n";
      //         cout << "RidgeletSolver took: " << nc_solve / 1e9 << " Gcycles\n";
      //         cout << "cg::relres: " << rt_solver.relres() << "\n";
      //         cout << "cg::iter: " << rt_solver.iter() << "\n\n";
      //       }
      // write arrays to hdf5
      complex_array_t tmp(Ny, Nx);
      rt.irt(tmp, x.coeffs());
      array_t sol(f * Ny, f * Nx);
      complex_array_t solh(f * Ny, f * Nx);
      solh.setZero();
      ftcut(solh, Ny / 2, Nx / 2) = ftcut(tmp, Ny / 2, Nx / 2);
      // ftcut(solh, Ny, Nx) = tmp;
      fft.ift(sol, solh);
      char buf[256];
      std::sprintf(buf, "%d_%d", qx, qy);
#pragma omp critical
      {
        hid_t group = H5Gcreate(file, buf, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        eigen2hdf::save(group, "sol", sol);
        eigen2hdf::save(group, "solh_ftcut", solh);
        eigen2hdf::save(group, "solh", tmp);
        H5Gclose(group);
      }
    }
  }

  eigen2hdf::save(file, "cg_relres", RELRES_MAT);
  eigen2hdf::save(file, "cg_iter", ITER_MAT);
  eigen2hdf::save(file, "cg_time", TIME_MAT);

  H5Fclose(file);

  return 0;
}
