#include <cstdio>

#include <base/logger.hpp>
#include <spectral/quadrature/qhermite.hpp>

#include "../advection_inflow.hpp"

int main(int argc, char* argv[])
{
  SOURCE_INFO();

  unsigned int Jx, Jy, rho_x, rho_y, K;
  std::string inith5;
  double dt;
  unsigned int gf;  // grid out factor
  double sigma;     // absorption strength
  int cg_maxit;
  double cg_tol;
  int ntsteps;

  po::options_description options("options");
  options.add_options()("help", "produce help message")
      ("Jx,i", po::value<unsigned int>(&Jx)->default_value(3), "Jx")
      ("Jy,j", po::value<unsigned int>(&Jy)->default_value(3), "Jy")
      ("rx,x", po::value<unsigned int>(&rho_x)->default_value(1), "rho_x")
      ("ry,y", po::value<unsigned int>(&rho_y)->default_value(1), "rho_x")
      ("init,I", po::value<std::string>(&inith5)->required(), "init file (must match K)")
      ("deg,K", po::value<unsigned int>(&K)->required(), "polynomial deg.")
      ("dt,t", po::value<double>(&dt)->default_value(0.1), "dt")
      ("nt,n", po::value<int>(&ntsteps)->default_value(1), "num. steps")
      ("sigma", po::value<double>(&sigma)->default_value(30), "absorption strength")
      ("sigmaf", po::value<std::string>()->default_value("const"), "sigma fct = [const|lin|smooth]")
      ("cg_tol", po::value<double>(&cg_tol)->default_value(1e-6), "cg tolerance")
      ("cg_maxit", po::value<int>(&cg_maxit)->default_value(400), "cg max. iterations")
      ("cg_log", "log cg history")
      ("save", "save solution");
  po::variables_map vm;

  try {
    po::store(po::parse_command_line(argc, argv, options), vm);
    if (vm.count("help")) {
      std::cout << options << "\n";
      return 0;
    }
    po::notify(vm);
  } catch (std::exception& e) {
    cout << e.what() << "\n";
    return 1;
  }

  cout << setw(20) << "Jx"
       << ": " << Jx << "\n"
       << setw(20) << "Jy"
       << ": " << Jy << "\n"
       << setw(20) << "rho_x"
       << ": " << rho_x << "\n"
       << setw(20) << "rho_y"
       << ": " << rho_y << "\n";

  boltzmann::QHermite qherm(0.5, K);

  hid_t ifile = H5Fopen(inith5.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  array_t qleft(K, K);
  array_t qtop(K, K);
  array_t rho(K, K);
  try {
    eigen2hdf::load(ifile, "qleft", qleft);
    eigen2hdf::load(ifile, "qtop", qtop);
    eigen2hdf::load(ifile, "rho", rho);
  } catch (...) {
  }

  RidgeletInflowTest ridgelet_inflow(vm);

  if (vm.count("cg_log")) ridgelet_inflow.log_cg(true);

  auto& v = qherm.pts();
  auto& log = Logger::GetInstance();

  Eigen::MatrixXd R(K, K);
  Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic> D(K, K);

  for (int i = 0; i < K; ++i) {
    for (int j = 0; j < K; ++j) {
      double vy = v[i];
      double vx = v[j];
      double mqleft = qleft(i, j);
      double mqtop = qtop(i, j);
      double f0 = rho(i, j);
      ridgelet_inflow.run(vx, vy, mqleft, mqtop, f0, ntsteps);
      double relres = ridgelet_inflow.solver().relres();
      int iter = ridgelet_inflow.solver().iter();
      char buf[256];
      sprintf(buf, "%d %d CG::SOLVER::relres %.5e", i, j, relres);
      log << buf;
      sprintf(buf, "%d %d CG::SOLVER::iter %d", i, j, iter);
      log << buf;

      R(i, j) = relres;
      D(i, j) = iter;
    }
  }

  {
    std::ofstream fout("advection_inflow_dir_relres.dat");
    fout << R;
  }
  {
    std::ofstream fout("advection_inflow_dir_iter.dat");
    fout << D;
  }

  return 0;
}
