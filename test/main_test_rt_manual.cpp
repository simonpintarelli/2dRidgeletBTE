// system includes -----------------------------------------------
#include <boost/math/constants/constants.hpp>
#include <boost/program_options.hpp>
#include <iostream>
// own includes --------------------------------------------------
#include <base/eigen2hdf.hpp>
#include <base/init.hpp>
#include <fft/fft2.hpp>
#include <ridgelet/ridgelet_cell_array.hpp>
#include <ridgelet/ridgelet_frame.hpp>
#include <ridgelet/rt.hpp>

using namespace std;

const char* fname = "test_rt.h5";

typedef RT<> RT_t;

typedef RT_t::array_t array_t;
typedef RT_t::complex_array_t complex_array_t;
typedef RT_t::rt_coeff_t rt_coeff_t;

void dump_frc(const std::vector<rt_coeff_t>& f_rc,
              const RidgeletFrame& rt,
              string fname = "f_rc.h5")
{
  hid_t file = H5Fcreate(fname.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  for (unsigned int i = 0; i < f_rc.size(); ++i) {
    stringstream ss;
    ss << rt.lambdas()[i];
    string slam = ss.str();
    eigen2hdf::save(file, slam, f_rc[i]);
  }
  H5Fclose(file);
  cout << "\n\n ---- Written f(lambda, t) to `" << fname << "` ----\n\n";
}

int main(int argc, char* argv[])
{
  SOURCE_INFO();

  namespace po = boost::program_options;

  cout << "FFTW_BACKWARD: " << FFTW_BACKWARD << "\n";
  cout << "FFTW_FORWARD: " << FFTW_FORWARD << "\n";

  unsigned int Jx, Jy, rho_x, rho_y;

  po::options_description options("options");
  options.add_options()("help", "produce help message")
      ("Jx,i", po::value<unsigned int>(&Jx)->default_value(2), "Jx")
      ("Jy,j", po::value<unsigned int>(&Jy)->default_value(2), "Jy")
      ("rx,x", po::value<unsigned int>(&rho_x)->default_value(1), "rho_x")
      ("ry,y", po::value<unsigned int>(&rho_y)->default_value(1), "rho_x");
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

  RidgeletFrame frame(Jx, Jy, rho_x, rho_y);
  const unsigned int ncols = frame.Nx();  // #cols
  const unsigned int nrows = frame.Ny();  // #rows
  // FFT fft;
  RT_t rt(frame);
  complex_array_t Fh(nrows, ncols);
  complex_array_t Fh2(nrows, ncols);
  /* compute ridgelet transform  */
  std::vector<rt_coeff_t> rt_coeffs(frame.size());
  std::vector<rt_coeff_t> rt_coeffs2(frame.size());
  rt.rt(rt_coeffs, Fh);

  for (unsigned int i = 0; i < frame.size(); ++i) {
    rt_coeffs[i].setZero();
  }
  bool failed = false;

  cout << "Set single lambda rt coeff to random and check invertibility..."
       << "\n";
  for (unsigned int i = 0; i < frame.size(); ++i) {
    rt_coeffs[i].setRandom();
    rt.irt(Fh, rt_coeffs);
    // forward transform
    rt.rt(rt_coeffs2, Fh);
    // inverse transform
    rt.irt(Fh2, rt_coeffs2);
    // check that is invertible on half size
    auto Diff = ftcut(Fh2, nrows / 2, ncols / 2) - ftcut(Fh, nrows / 2, ncols / 2);
    double diff = Diff.abs().sum();
    if (diff > 1e-11) {
      cout << "::check inverse::" << frame.lambdas()[i] << ": " << diff << "\n";
      failed = true;
    }
  }

  if (!failed)
    cout << "TEST PASSSED"
         << "\n";
  else
    cout << "TEST FAILED"
         << "\n";

  return 0;
}
