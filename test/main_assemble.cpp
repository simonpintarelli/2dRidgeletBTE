#include <base/eigen2hdf.hpp>
#include <base/init.hpp>
#include <base/timer.hpp>
#include <boost/program_options.hpp>
#include <ridgelet/construction/ft.hpp>
#include <ridgelet/ridgelet_frame.hpp>
#include "ridgelet/basis.hpp"

#include <fstream>
#include <iostream>
#include <sstream>

using namespace std;

void dump(const RidgeletFrame& RF)
{
  const char* fname = "assemble.h5";
  hid_t file = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  auto& lambdas = RF.lambdas();

  for (auto& lam : lambdas) {
    if (lam.t == rt_type::S) {
      auto& M = RF.get_dense(lam);
      stringstream ss;
      ss << lam;
      string dset = ss.str();
      //      cout << "creating dset: " << dset << "\n";
      eigen2hdf::save(file, dset, M);
    } else {
      auto& M = RF.get_sparse(lam);
      stringstream ss;
      ss << lam;
      string dset = ss.str();
      //      cout << "creating dset:" << dset << "\n";
      eigen2hdf::save_sparse(file, dset, M);
    }
  }
  H5Fclose(file);
  cout << "wrote ridgelet coeffs to " << fname << "\n";
}

void listb(const RidgeletFrame& RF, unsigned int rho_x, unsigned int rho_y)
{
  for (auto& lam : RF.lambdas()) {
    unsigned int tx = translation_size(lam.t, lam.j, rho_x, 0);
    unsigned int ty = translation_size(lam.t, lam.j, rho_y, 1);

    if (lam.t == rt_type::S) {
      auto& M = RF.get_dense(lam);
      stringstream ss;
      ss << lam;
      string sl = ss.str();
      stringstream dim;
      dim << M.rows() << " x " << M.cols();
      cout << setw(15) << sl << ", rt_size: " << setw(15) << dim.str() << ", T: " << setw(5) << tx
           << " x " << setw(5) << ty << "\n";
    } else {
      auto& M = RF.get_sparse(lam);
      stringstream ss;
      ss << lam;
      string sl = ss.str();
      stringstream dim;
      dim << M.rows() << " x " << M.cols();
      cout << setw(15) << sl << ", rt_size: " << setw(15) << dim.str() << ", T: " << setw(5) << tx
           << " x " << setw(5) << ty << "\n";
    }
  }
  cout << "T: translation set dimensions, rt_size: ridgelet coefficient array dimensions"
       << "\n";
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
      ("rx,x", po::value<unsigned int>(&rho_x)->default_value(1), "rho_x")
      ("ry,y", po::value<unsigned int>(&rho_y)->default_value(1), "rho_x")
      ("dump", "dump coefficients to hdf5")
      ("list", "list basis and sizes");
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
  auto tlap = timer.stop();
  cout << "RidgeletFrame::RidgeletFrame(): " << tlap / 1e9 << " [Gcycles]"
       << "\n";

  cout << "frame.size: " << frame.size() << "\n";
  cout << "frame.Nx: " << frame.Nx() << "\n";
  cout << "frame.Ny: " << frame.Ny() << "\n";

  {
    ofstream fout("assemble_rt.txt");
    for (auto& lam : frame.lambdas()) {
      fout << lam << endl;
    }
    fout.close();
  }

  if (vm.count("dump")) dump(frame);

  if (vm.count("list")) listb(frame, rho_x, rho_y);

  return 0;
}
