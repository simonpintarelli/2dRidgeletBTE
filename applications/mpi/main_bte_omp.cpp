// system includes -----------------------------------------------
#include <mpi.h>
#include <algorithm>
#include <boost/lexical_cast.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/program_options.hpp>
#include <iostream>
#include "base/exceptions.hpp"
#include "base/logger.hpp"
#include "base/pcout.hpp"
#include "export/output_handler.hpp"
#ifdef USE_OMP
#include <omp.h>
#endif

// own includes --------------------------------------------------
#include "base/coeff_array.hpp"
#include "base/eigen2hdf.hpp"
#include "base/import_coeffs.hpp"
#include "base/init.hpp"
#include "base/numbers.hpp"
#include "fft/fft2_r2c.hpp"
#include "ridgelet/ridgelet_cell_array.hpp"
#include "ridgelet/ridgelet_frame.hpp"
#include "ridgelet/rt.hpp"
#ifdef FTCG
#include "ridgelet/rc_linearize.hpp"
#endif
#include "operators/operators.hpp"
#include "solver/cg.hpp"
#include "solver/ridgelet_solver.hpp"
// boltzmann includes --------------------------------------------
#include "spectral/basis/spectral_basis_factory_ks.hpp"
#include "spectral/collision/collision_tensor_galerkin.hpp"
#include "spectral/polar_to_nodal.hpp"

using namespace std;
namespace po = boost::program_options;

typedef boltzmann::SpectralBasisFactoryKS basis_factory_t;
typedef basis_factory_t::basis_type basis_type;
typedef RT<double, RidgeletFrame, FFTr2c<PlannerR2C> > RT_t;
typedef RT_t::array_t array_t;
typedef RT_t::complex_array_t complex_array_t;
typedef RT_t::rt_coeff_t rt_coeff_t;

const double Lx = 1;
const double Ly = 1;

class BTE
{
 public:
  BTE(const po::variables_map& vm)
  {
    unsigned int Jx = vm["Jx"].as<unsigned int>();
    unsigned int Jy = vm["Jy"].as<unsigned int>();
    unsigned int rho_x = vm["rx"].as<unsigned int>();
    unsigned int rho_y = vm["ry"].as<unsigned int>();

    // make sure that Lx = 2**j
    rt_frame = RidgeletFrame(Jx - 1, Jy - 1, rho_x, rho_y);

    load_spectral_basis("spectral_basis.desc");

    K = spectral::get_max_k(spectral_basis) + 1;
    Nx = rt_frame.Nx();  // #cols
    Ny = rt_frame.Ny();  // #rows
    Lx = Nx / 2;
    Ly = Ny / 2;
    dimX = Lx * Ly;  // num physical DoFs
    dimV = K * K;    // num DoFs in spectral basis (nodal)

    xmap = Map(dimX);
    vmap = Map(dimV);

    ASSERT(Nx > 0);
    ASSERT(Ny > 0);
    ASSERT(K > 0);
    ASSERT(dimV > 0);
    ASSERT(dimX > 0);
  }

 private:
  void load_spectral_basis(const std::string& fname)
  {
    basis_factory_t::create(spectral_basis, fname);
  }

 public:
  RidgeletFrame rt_frame;
  basis_type spectral_basis;
  unsigned int Nx = 0;
  unsigned int Ny = 0;
  unsigned int Lx = 0;
  unsigned int Ly = 0;
  unsigned int K = 0;
  unsigned int dimV = 0;
  unsigned int dimX = 0;

  Map xmap;
  Map vmap;
};

struct bprogram_input
{
  static unsigned int Jx, Jy, rho_x, rho_y;
  static double dt;
  static unsigned int ntsteps;
  static unsigned int obuf, of;
  static int dump_freq;
  static double knudsen;
#ifdef FTCG
  static double rttre;
#else
  static double cg_tol;
  static int cg_maxit;
#endif

  static void print()
  {
#ifdef FTCG
    ASSERT(rttre <= 1);
#endif
    pcout << setw(20) << "Jx"
          << ": " << Jx << "\n"
          << setw(20) << "Jy"
          << ": " << Jy << "\n"
          << setw(20) << "rho_x"
          << ": " << rho_x << "\n"
          << setw(20) << "rho_y"
          << ": " << rho_y << "\n"
          << setw(20) << "dt"
          << ": " << dt << "\n"
          << "\n";
#ifndef FTCG
    pcout << setw(20) << "cg_tol"
          << ": " << cg_tol << "\n"
          << setw(20) << "cg_maxit"
          << ": " << cg_maxit << "\n";
#endif
    pcout << setw(20) << "obuf:"
          << ":" << obuf << "\n";
  }
};

unsigned int bprogram_input::Jx, bprogram_input::Jy, bprogram_input::rho_x, bprogram_input::rho_y;
unsigned int bprogram_input::ntsteps, bprogram_input::obuf, bprogram_input::of;
int bprogram_input::dump_freq;
double bprogram_input::dt, bprogram_input::knudsen;
#ifdef FTCG
double bprogram_input::rttre;
#else
double bprogram_input::cg_tol;
int bprogram_input::cg_maxit;
#endif

int main(int argc, char* argv[])
{
  int mpi_provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &mpi_provided);
  ASSERT(mpi_provided = MPI_THREAD_SERIALIZED);

  int pid = -1, nprocs = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  ASSERT(pid >= 0);
  ASSERT(nprocs > 0);

  pcout << "SOURCE::INFO::GIT_BRANCHNAME " << GIT_BNAME << "\n";
  pcout << "SOURCE::INFO::GIT_SHA1       " << GIT_SHA1 << "\n";
#ifdef FTCG
  pcout << "use FFT CG solver"
        << "\n";
#endif

  po::options_description options("options");
  options.add_options()("help", "produce help message")
      ("Jx,i", po::value<unsigned int>(&bprogram_input::Jx)->default_value(3), "Jx (corresponds to Lx=2**(Jx+1)*rx)")
      ("Jy,j", po::value<unsigned int>(&bprogram_input::Jy)->default_value(3), "Jy (corresponds to Ly=2**(Jy+1)*ry)")
      ("rx,x", po::value<unsigned int>(&bprogram_input::rho_x)->default_value(1), "rho_x")
      ("ry,y", po::value<unsigned int>(&bprogram_input::rho_y)->default_value(1), "rho_x")
      ("f0", po::value<string>()->default_value(""), "initial distribution, dset='coeffs' (Polar basis)")
      ("init", po::value<string>()->required(), "initial distribution as matrix of size L x N (nodal basis of total size N)")
      ("dt,t", po::value<double>(&bprogram_input::dt)->default_value(0.001), "timesteps")
      ("nt,n", po::value<unsigned int>(&bprogram_input::ntsteps)->default_value(1000), "num timesteps")
      ("kn", po::value<double>(&bprogram_input::knudsen)->default_value(1.0), "Knudsen number")
      ("obuf", po::value<unsigned int>(&bprogram_input::obuf)->default_value(1), "buffer num timesteps")
      ("of", po::value<unsigned int>(&bprogram_input::of)->default_value(1), "output frequency")
      ("dump", po::value<int>(&bprogram_input::dump_freq)->default_value(-1), "dump full solution vector every n timesteps")
#ifdef FTCG
      ("rttre", po::value<double>(&bprogram_input::rttre)->default_value(1), "keep rttre rt-coefficients in every timestep")
#else
      ("cg_maxit", po::value<int>(&bprogram_input::cg_maxit)->default_value(100), "cg max. iterations")
      ("cg_tol", po::value<double>(&bprogram_input::cg_tol)->default_value(1e-6), "cg rel. tol.")
#endif
      ("solverlog", "log convergence details of cg solver")
      ("no-collision", "disable collision")
      ("export-dofs", "export grid information");
  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, options), vm);
    po::notify(vm);
  } catch (std::exception& e) {
    if (vm.count("help"))
      pcout << options << "\n";
    else
      pcout << e.what() << "\n";
    return 1;
  }

  // write parameters to stdout
  bprogram_input::print();

  if (vm["f0"].as<std::string>() != "" && vm["init"].as<std::string>() != "") {
    cout << "cannot specify --f0 and --init at the same time"
         << "\n";
    MPI_Finalize();
    return 1;
  }

  if (vm.count("help")) {
    std::cout << options << "\n";
    MPI_Finalize();
    return 0;
  }

  BTE main_class(vm);

  auto& rf = main_class.rt_frame;
  const int Nx = main_class.Nx;   // #cols
  const int Ny = main_class.Ny;   // #rows
  const int L = main_class.dimX;  // num physical DoFs
  const int K = main_class.K;
  // const int N = main_class.dimV; // num DoFs in spectral basis (nodal)
  // Nx is 2 time Lx
  pcout << "Nx: " << Nx << "\n";
  pcout << "Ny: " << Ny << "\n";

  if (vm.count("export-dofs")) {
    if (pid == 0) {
      hid_t file = H5Fcreate("grid.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
      Eigen::ArrayXd xi = Eigen::ArrayXd::LinSpaced(Nx + 1, 0, Lx).segment(0, Nx);
      Eigen::ArrayXd yi = Eigen::ArrayXd::LinSpaced(Ny + 1, 0, Ly).segment(0, Ny);
      eigen2hdf::save(file, "xi", xi);
      eigen2hdf::save(file, "xi", yi);
      H5Fclose(file);
    }
    MPI_Finalize();
    return 0;
  }

  RT_t rt(rf);
  FFTr2c<PlannerR2C> fft;
  init_fftw(fft, FFTW_MEASURE, rf);
  typedef RidgeletCellArray<rt_coeff_t> rca_t;

  // ======================================
  // create rte_map / vector
  // transport problem is distributed in V
  CoeffArray<double> rte_vector(main_class.vmap, main_class.dimX);
  // collision problem is distributed in X
  CoeffArray<double> bte_vector(main_class.xmap, main_class.dimV);

  // ========================================
  // ========= Polar Spectral setup =========
  // ========================================
  auto& spectral_basis = main_class.spectral_basis;
  const int Np = spectral_basis.n_dofs();
  boltzmann::Polar2Nodal<basis_type> p2n;
  p2n.init(spectral_basis, 1.0);

  // ------------------------------------------------------------
  // initialize coefficients at t=0
  // ------------------------------------------------------------
  load_coeffs_from_file(rte_vector, vm["init"].as<std::string>(), "coeffs");
  bte_vector = rte_vector.transpose();

  // load collision tensor
  boltzmann::CollisionTensorGalerkin collision_operator(spectral_basis);
  if (!vm.count("no-collision")) collision_operator.read_hdf5("collision_tensor.h5");
  pcout << "done loading collision tensor\n";

  std::vector<double> qi(K);
  boltzmann::gauss_hermite_roots(qi, K);

  OutputHandler output_handler(spectral_basis,
                               main_class.xmap,
                               p2n,
                               bprogram_input::of,         // output frequency for moments
                               bprogram_input::dump_freq,  // output frequency for full solution
                               bprogram_input::obuf);
  output_handler.compute(bte_vector, 0, 0.);

// ======================================== //
// ===========  time STEPPING ============= //
// ======================================== //
#ifdef FTCG
  RCLinearize rcl(rf);
  unsigned int nc = rcl.size();
  std::vector<int> active_set(nc, 0);
#endif

#pragma omp parallel
  {
    // --- thread local variables --- //
    double current_time = 0;
    auto v_ldofs = main_class.vmap.lsize();
    auto x_ldofs = main_class.xmap.lsize();
    Eigen::VectorXd cp(Np);
    Eigen::VectorXd cp_out(Np);
    Eigen::MatrixXd cn(K, K);
    array_t buf_rte(Ny / 2,
                    Nx / 2);
    // variables transport
    complex_array_t Bh(Ny, Nx);
    complex_array_t Fh(Ny, Nx);
    complex_array_t Fh_cut(Ny / 2, Nx / 2);
    rca_t f_rc(main_class.rt_frame);
    rca_t b_rc(main_class.rt_frame);
    double* rte_data = rte_vector.data();

#ifdef FTCG
    std::vector<double> lin_f_rc(rcl.size());
#endif

    for (unsigned int ii = 1; ii <= bprogram_input::ntsteps; ++ii) {
#pragma omp single
      pcout << "timestep " << ii << "\n";

// ========== solve transport problem==========
#pragma omp for schedule(static)
      for (int ijv = 0; ijv < v_ldofs; ++ijv) {
        Eigen::Map<array_t> rte_view(rte_data + ijv * L, Ny / 2, Nx / 2);
        buf_rte = rte_view;
        fft.ft(Fh_cut, buf_rte, false);
        Fh.setZero();
        ftcut(Fh, Ny / 2, Ny / 2) = Fh_cut;
        hf_zero(Fh);
        const int jv = main_class.vmap.GID(ijv);  // global V-index

        // polar2nodal assumes col-major storage
        const double vx = qi[jv / K];
        const double vy = qi[jv % K];

        // transport operator
        TransportOperator T(vx, vy, Lx, Ly, Nx, Ny, bprogram_input::dt);
#ifdef FTCG
        T.apply_bckwrd_euler(Fh);
        if (bprogram_input::rttre < 1) {
          rt.rt(f_rc.coeffs(), Fh);
          lin_f_rc = rcl.linearize(f_rc.coeffs());
          double tre = std::max(rcl.get_threshold(lin_f_rc, bprogram_input::rttre), 1e-12);
          // apply threshold
          rcl.threshold(f_rc, tre);

          // update active set
          for (unsigned int i = 0; i < rcl.size(); ++i) {
            // synchronization is not needed!
            if (std::abs(lin_f_rc[i]) > tre) active_set[i] = 1;
          }
        }
#else
        AhAOp AhA(vx, vy, Lx, Ly, Nx, Ny, bprogram_input::dt);
        // * solve with ridgelets * //
        PTransportOp<RT_t> A(rt, AhA, vx, vy);
        // init x0
        // initialize starting value for cg
        rt.rt(f_rc.coeffs(), Fh);
        // create rhs
        rt.rt(b_rc.coeffs(), Bh);
        RidgeletSolver<rt_coeff_t> rt_solver(rf, vx, vy);
        rt_solver.solve(f_rc, A, b_rc, bprogram_input::cg_tol, bprogram_input::cg_maxit);
#endif
#ifdef FTCG
        if (bprogram_input::rttre < 1) {
          rt.irt(Fh, f_rc.coeffs());  // go back to Fourier domain
        }
#else
        rt.irt(Fh, f_rc.coeffs());  // go back to Fourier domain
#endif

        Fh_cut = ftcut(Fh, Ny / 2, Nx / 2);
        fft.ift(buf_rte, Fh_cut);  // go back to Spatial domain
        rte_view = buf_rte;
      }                            // end for velocities, OMP => implied omp barrier
#pragma omp single
      {
// reduce active set to proc 0 and print statistics...
#ifdef FTCG
        std::vector<int> active_set_root(rcl.size(), 0);
        int ierr = MPI_Reduce(active_set.data(),
                              active_set_root.data(),
                              rcl.size(),
                              MPI_INT,
                              MPI_LOR,
                              0,
                              MPI_COMM_WORLD);
        BOOST_ASSERT(ierr == 0);
        if (pid == 0) {
          active_set = active_set_root;
          int ac = 0;
          for (unsigned q = 0; q < active_set.size(); ++q) {
            if (active_set[q]) ac++;
          }
          std::printf("ACTIVE_SIZE::frame=%d\t%d\n", ii, ac);
        }
#endif
        // communicate coefficients
        bte_vector = rte_vector.transpose();
      }
#pragma omp for schedule(static)
      for (int ijx = 0; ijx < x_ldofs; ++ijx) {
        // 1. transform to polar basis
        Eigen::Map<Eigen::MatrixXd> cn(bte_vector.get(ijx).data(), K, K);
        p2n.to_polar(cp, cn);
        if (vm.count("no-collision")) {
          // do nothing
        } else {
          // 2. apply Q
          collision_operator.apply(cp_out.data(), cp.data());
          // explicit euler
          cp_out = cp_out * bprogram_input::dt / bprogram_input::knudsen + cp;
          collision_operator.project(cp_out.data(), cp.data());
          // 3. transform to lagrange basis
          p2n.to_nodal(cn, cp_out);
          // output
        }
      }
      current_time += bprogram_input::dt;

#pragma omp master
      {
        output_handler.compute(bte_vector, ii, current_time);
        rte_vector = bte_vector.transpose();
      }
#pragma omp barrier
    }  // end timesteps
#ifdef FTCG
#pragma omp single
    {
      if (pid == 0) {
        // write active_set to disk...
        std::ofstream fout("ac.coeff.final.data");
        for (unsigned int i = 0; i < active_set.size(); ++i) {
          fout << active_set[i] << std::endl;
        }
        fout.close();
      }
    }
#endif
  }  // end parallel region

  MPI_Finalize();
  return 0;
}
