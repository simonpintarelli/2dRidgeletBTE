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
#include <fft/planner.hpp>
#include <ridgelet/ridgelet_frame.hpp>
#include <ridgelet/rt.hpp>

using namespace std;

const char* fname = "approximation.h5";

typedef RT<std::complex<double>, RidgeletFrame, FFTr2c<PlannerR2C> > RT_t;

typedef RT_t::array_t array_t;
typedef RT_t::complex_array_t complex_array_t;
typedef RT_t::rt_coeff_t rt_coeff_t;

void dump_frc(const std::vector<rt_coeff_t>& f_rc, const RidgeletFrame& rt)
{
  const char* fname = "f_rc_random.h5";
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

int main(int argc, char* argv[])
{
  SOURCE_INFO();

  namespace po = boost::program_options;

  unsigned int Jx, Jy, rho_x, rho_y;
  int Jt;

  po::options_description options("options");
  options.add_options()("help", "produce help message")
      ("Jx,i", po::value<unsigned int>(&Jx)->default_value(6), "Jx")
      ("Jy,j", po::value<unsigned int>(&Jy)->default_value(6), "Jy")
      ("rx,x", po::value<unsigned int>(&rho_x)->default_value(1), "rho_x")
      ("ry,y", po::value<unsigned int>(&rho_y)->default_value(1), "rho_x")
      ("input,f", po::value<std::string>(), "input file")
      ("Jt", po::value<int>(&Jt)->default_value(-1), "truncate at level Jt");
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

  RDTSCTimer timer;

  timer.start();
  RidgeletFrame frame(Jx, Jy, rho_x, rho_y);
  auto t_create_frame = timer.stop();
  cout << "TIMINGS::create_frame: " << t_create_frame / 1e9 << "\n";

  const unsigned int ncols = frame.Nx();  // #cols
  const unsigned int nrows = frame.Ny();  // #rows
  cout << setw(20) << "Ny"
       << ":" << nrows << "\n";
  cout << setw(20) << "Nx"
       << ":" << ncols << "\n";

  RT_t rt(frame);
  array_t F(nrows, ncols);
  array_t Fo(nrows, ncols);

  if (!vm.count("input")) {
    cout << "generating random input"
         << "\n";
    F.setRandom();
    F *= nrows * ncols;
  } else {
    std::string fname = vm["input"].as<std::string>();
    cout << "read input from file: " << fname << "\n";
    hid_t lfile = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    eigen2hdf::load(lfile, "R", F);
    // matlab is column major ...
    F = F.transpose();
    Fo = F;
  }
  typename RT_t::fft_t fft;
  timer.start();
  init_fftw(fft, FFTW_ESTIMATE, frame);
  timer.print(cout, timer.stop(), "FFTW plans");

  complex_array_t Fh(F.rows(), F.cols());
  fft.ft(Fh, F);
  std::vector<rt_coeff_t> rt_coeffs(frame.size());
  rt.rt(rt_coeffs, Fh);
  // dump_frc(rt_coeffs, frame);
  auto& Lambda = frame.lambdas();
  for (unsigned int i = 0; i < rt_coeffs.size(); ++i) {
    if (Lambda[i].j != -1 && Lambda[i].j >= Jt) rt_coeffs[i].setZero();
  }

  // -------------------- Inverse transform --------------------
  complex_array_t Fh2(nrows, ncols);
  rt.irt(Fh2, rt_coeffs);
  hid_t file = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  eigen2hdf::save(file, "Fh", Fh);
  eigen2hdf::save(file, "R", Fo);
  eigen2hdf::save(file, "Fh2", Fh2);
  H5Fclose(file);
  cout << "written results to " << fname << "\n";

  complex_array_t diff = (ftcut(Fh2, nrows / 2, ncols / 2) - ftcut(Fh, nrows / 2, ncols / 2));
  cout << "avg diff (ft): " << diff.abs().sum() / (nrows / 2 * ncols / 2) << "\n";

  array_t Fr(nrows / 2, ncols / 2);
  array_t Fr2(nrows / 2, ncols / 2);

  cout << "nrows/2 * ncols/2: " << (nrows / 2 * ncols / 2) << "\n";

  fft.ift(Fr, ftcut(Fh, nrows / 2, ncols / 2));
  fft.ift(Fr2, ftcut(Fh2, nrows / 2, ncols / 2));

  array_t diffr = Fr - Fr2;
  cout << "avg diff (real):" << diffr.abs().sum() / (nrows / 2 * ncols / 2) << "\n";

  return 0;
}
