#include <cstdio>
#include <iostream>

#include "base/eigen2hdf.hpp"
#include "base/init.hpp"
#include "base/timer.hpp"
#include "fft/fft2_r2c.hpp"
#include "fft/planner.hpp"
#include "ridgelet/ridgelet_frame.hpp"
#include "ridgelet/rt.hpp"

#include <boost/program_options.hpp>
#include <iomanip>

using namespace std;

const char* fname = "test_fft2.h5";

int main(int argc, char* argv[])
{
  SOURCE_INFO();

  namespace po = boost::program_options;

  unsigned int Jx, Jy, rho_x, rho_y;

  po::options_description options("options");
  options.add_options()("help", "produce help message")
      ("Jx,i", po::value<unsigned int>(&Jx)->default_value(3), "Jx")
      ("Jy,j", po::value<unsigned int>(&Jy)->default_value(3), "Jy")
      ("rx,x", po::value<unsigned int>(&rho_x)->default_value(10), "rho_x")
      ("ry,y", po::value<unsigned int>(&rho_y)->default_value(10), "rho_x");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, options), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << options << "\n";
    return 0;
  }
  cout << setw(20) << "Jx"
       << ": " << Jx << "\n"
       << setw(20) << "Jy"
       << ": " << Jy << "\n"
       << setw(20) << "rho_x"
       << ": " << rho_x << "\n"
       << setw(20) << "rho_y"
       << ": " << rho_y << "\n";

  FFTr2c<PlannerR2C> fft;

  RDTSCTimer timer;
  timer.start();
  RidgeletFrame frame(Jx, Jy, rho_x, rho_y);
  auto tlap = timer.stop();
  cout << "RidgeletFrame::RidgeletFrame(): " << tlap / 1e9 << " [Gcycles]"
       << "\n";

  timer.start();
  init_fftw(fft, FFTW_MEASURE, frame);
  auto tlap_fftw_plans = timer.stop();
  timer.print(cout, tlap_fftw_plans, "fftw plans");

  unsigned int Nx = frame.Nx();
  unsigned int Ny = frame.Ny();

  cout << "Nx: " << Nx
       << "\n"
       << "Ny: " << Ny
       << "\n";

  return 0;
}
