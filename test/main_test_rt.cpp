// system includes -----------------------------------------------
#include <boost/math/constants/constants.hpp>
#include <boost/program_options.hpp>
#include <iostream>
// own includes --------------------------------------------------
#include <base/eigen2hdf.hpp>
#include <base/init.hpp>
#include <base/timer.hpp>
#include <fft/fft2.hpp>
#include <ridgelet/ridgelet_frame.hpp>
#include <ridgelet/rt.hpp>

using namespace std;

const char* fname = "test_rt.h5";

// typedef FFTr2c<PlannerR2COD> fft_t;
typedef FFT fft_t;
typedef RT<std::complex<double>, RidgeletFrame, fft_t> RT_t;
typedef RT_t::array_t array_t;
typedef RT_t::complex_array_t complex_array_t;
typedef RT_t::rt_coeff_t rt_coeff_t;

void dump_frc(const std::vector<rt_coeff_t>& f_rc, const RidgeletFrame& rt)
{
  const char* fname = "f_rc.h5";
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

  po::options_description options("options");
  options.add_options()
      ("help", "produce help message")
      ("Jx,i", po::value<unsigned int>(&Jx)->default_value(3), "Jx")
      ("Jy,j", po::value<unsigned int>(&Jy)->default_value(3), "Jy")
      ("rx,x", po::value<unsigned int>(&rho_x)->default_value(1), "rho_x")
      ("ry,y", po::value<unsigned int>(&rho_y)->default_value(1), "rho_x")
      ("save", "save coefficients");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, options), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << options << "\n";
    return 0;
  }

  cout << setw(20) << "Jx: " << Jx << "\n"
       << setw(20) << "Jy: " << Jy << "\n"
       << setw(20) << "rho_x: " << rho_x << "\n"
       << setw(20) << "rho_y: " << rho_y << "\n"
       << setw(20) << "Nx: " << std::pow(2, Jx + 2) * rho_x << "\n"
       << setw(20) << "Ny: " << std::pow(2, Jy + 2) * rho_y << "\n";
  RDTSCTimer timer;

  timer.start();
  RidgeletFrame frame(Jx, Jy, rho_x, rho_y);
  double time_frame_constructor = timer.stop();
  timer.print(cout, time_frame_constructor, "RidgeletFrame init");

  const unsigned int ncols = frame.Nx();  // #cols
  const unsigned int nrows = frame.Ny();  // #rows

  RT_t rt(frame);
  array_t F(nrows / 2, ncols / 2);
  {
    double pi = boost::math::constants::pi<double>();
    Eigen::ArrayXd x = pi * Eigen::ArrayXd::LinSpaced(ncols / 2, 0, 1);
    Eigen::ArrayXd y = pi * Eigen::ArrayXd::LinSpaced(nrows / 2, 0, 1);
    F = y.replicate(1, x.rows()).sin() * x.transpose().replicate(y.rows(), 1).sin();

    Eigen::MatrixXd tmp = F;
    tmp = tmp.triangularView<Eigen::Lower>();
    F = tmp.array();
  }

  fft_t fft;
  // debug
  complex_array_t Fhh(nrows / 2, ncols / 2);
  fft.ft(Fhh, F, false);
  hid_t file;
  if (vm.count("save")) {
    file = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    eigen2hdf::save(file, "Fhh", Fhh);  // debug
  }
  // ----------------------------------------
  complex_array_t Fh(nrows, ncols);
  Fh.setZero();
  ftcut(Fh, nrows / 2, ncols / 2) = Fhh;
  std::vector<rt_coeff_t> rt_coeffs(frame.size());
  timer.start();
  cout << "rt.rt(...)"
       << "\n";
  rt.rt(rt_coeffs, Fh);

  int ncoeffs = 0;
  for (unsigned int i = 0; i < rt_coeffs.size(); ++i) {
    ncoeffs += rt_coeffs[i].rows() * rt_coeffs[i].cols();
  }
  cout << "dim(rt_coeffs): " << ncoeffs << "\n";
  auto time_rt = timer.stop();
  timer.print(cout, time_rt, "rt.rt");

  // -------------------- Inverse transform --------------------
  complex_array_t Fh2(nrows, ncols);
  timer.start();
  cout << "rt.irt(...)"
       << "\n";
  rt.irt(Fh2, rt_coeffs);
  auto time_irt = timer.stop();
  timer.print(cout, time_irt, "rt.irt");

  array_t F2(nrows / 2, ncols / 2);
  complex_array_t Fh2_cut(nrows / 2, ncols / 2);
  Fh2_cut.setZero();
  Fh2_cut = ftcut(Fh2, nrows / 2, ncols / 2);
  fft.ift(F2, Fh2_cut);
  auto diff = (F - F2).abs();
  cout << "(F-F2).abs().sum(): " << diff.sum() << "\n";

  if (vm.count("save")) {
    eigen2hdf::save(file, "Fhl", Fhh);
    eigen2hdf::save(file, "Fh", Fh);
    eigen2hdf::save(file, "R", F);
    eigen2hdf::save(file, "Fh2", Fh2);
    eigen2hdf::save(file, "R2", F2);
    H5Fclose(file);
    cout << "written results to " << fname << "\n";
  }

  return 0;
}
