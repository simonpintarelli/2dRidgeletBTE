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

  RidgeletFrame rf(Jx, Jy, rho_x, rho_y);

  const unsigned int ncols = rf.Nx();  // #cols
  const unsigned int nrows = rf.Ny();  // #rows
  double pi = boost::math::constants::pi<double>();
  Eigen::ArrayXd x = pi * Eigen::ArrayXd::LinSpaced(ncols, 0, 1);
  Eigen::ArrayXd y = pi * Eigen::ArrayXd::LinSpaced(nrows, 0, 1);

  RT_t rt(rf);
  array_t F(nrows, ncols);
  F = y.replicate(1, x.rows()).sin() * x.transpose().replicate(y.rows(), 1).sin();
  // ----------------------------------------
  double n[2] = {1, 1};
  F = ((n[0] * (y.replicate(1, x.rows()) - 0.5 * pi) +
        n[1] * (x.transpose().replicate(y.rows(), 1) - 0.5 * pi)) > 1e-8)
          .select(F, array_t::Zero(nrows, ncols));
  cout << "Using non-smooth function"
       << "\n";

  FFT fft;
  // debug
  complex_array_t Fhh;
  fft.fft2(Fhh, F, false);
  hid_t file = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  eigen2hdf::save(file, "Fhh", Fhh);  // debug
  // ----------------------------------------
  complex_array_t Fh;
  fft.ft(Fh, F, false);
  RidgeletCellArray<rt_coeff_t> rt_cell_array(rf);
  auto& rt_coeffs = rt_cell_array.coeffs();

  rt.rt(rt_coeffs, Fh);
  dump_frc(rt_coeffs, rf);

  cout << "f_rc.norm() = " << rt_cell_array.norm() << endl;

  // -------------------- Inverse transform --------------------
  complex_array_t Fh2(nrows, ncols);
  rt.irt(Fh2, rt_coeffs);

  eigen2hdf::save(file, "Fh", Fh);
  eigen2hdf::save(file, "R", F);
  eigen2hdf::save(file, "Fh2", Fh2);
  H5Fclose(file);
  cout << "written results to " << fname << "\n";

  return 0;
}
