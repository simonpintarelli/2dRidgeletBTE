/**
 * @file   main_inflow_source.cpp
 * @author Simon Pintarelli <simon@thinkpadX1>
 * @date   Thu Apr 28 00:22:50 CEST 2016
 *
 * @brief  create sources in physical space and export to hdf
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

#include <boundary_conditions/inflow_bc.hpp>
#include <operators/operators.hpp>
#include <solver/cg.hpp>
#include <solver/ridgelet_solver.hpp>
#include <spectral/quadrature/gauss_hermite_roots.hpp>

using namespace std;
// typedef FFTr2c<PlannerR2C> fft_t;
// typedef RT<RidgeletFrame, fft_t> RT_t;
// typedef RT_t::array_t array_t;
// typedef RT_t::complex_array_t complex_array_t;
// typedef RT_t::rt_coeff_t rt_coeff_t;

const double Lx = 1.2;
const double Ly = 1.2;

// inflow values
const double ql = 0.5;
const double qt = 1.0;

int main(int argc, char *argv[])
{
  SOURCE_INFO();

  namespace po = boost::program_options;

  unsigned int N;
  unsigned int K;

  po::options_description options("options");
  options.add_options()("help", "produce help message")
      ("size,N", po::value<unsigned int>(&N)->default_value(128), "grid size")
      ("deg,K", po::value<unsigned int>(&K)->default_value(10), "deg");
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

  const unsigned int Nx = N;
  const unsigned int Ny = N;
  cout << "Nx: " << Nx << "\n";
  cout << "Ny: " << Ny << "\n";
  cout << "K:  " << K << "\n";
  cout << "ql: " << ql << "\n";
  cout << "qt: " << qt << "\n";
  // fft_t fft;
  // init_fftw(fft, FFTW_MEASURE, rf);
  // fft.get_plan().create_and_get_plan(f*Ny, f*Nx, PlannerR2C::INV);
  // fft.get_plan().create_and_get_plan(f*Ny, f*Nx, PlannerR2C::FWD);
  std::vector<double> qi(K);
  boltzmann::gauss_hermite_roots(qi, K);
  Eigen::ArrayXd xi = Eigen::ArrayXd::LinSpaced(Nx + 1, 0, Lx).segment(0, Nx);
  Eigen::ArrayXd yi = Eigen::ArrayXd::LinSpaced(Ny + 1, 0, Ly).segment(0, Ny);

  typedef Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> array_t;

  for (int i = 0; i < K; ++i) {
    cout << qi[i] << "\t";
  }
  cout << "\n";

  hid_t file = H5Fcreate("inflow_source.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  array_t X = xi.transpose().replicate(Ny, 1);
  array_t Y = yi.replicate(1, Nx);
#pragma omp parallel for
  for (int qx = 0; qx < K; ++qx) {
    for (int qy = 0; qy < K; ++qy) {
      const double vx = qi[qx];
      const double vy = qi[qy];
      array_t Q(Ny, Nx);
      make_inflow_source(Q, vx, vy, Lx, Ly, ql, qt);
      Q = ((X > 1 || Y > 1)).select(Q, array_t::Zero(Ny, Nx));

#pragma omp critical
      {
        eigen2hdf::save(
            file, boost::lexical_cast<string>(qx) + "_" + boost::lexical_cast<string>(qy), Q);
      }
    }
  }
  H5Fclose(file);

  return 0;
}
