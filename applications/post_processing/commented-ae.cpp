#include <hdf5.h>
#include <omp.h>
#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>
#include <algorithm>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <memory>
#include <regex>
#include <tuple>

#include "base/eigen2hdf.hpp"
#include "base/exceptions.hpp"
#include "brt_config.h"
#include "fft/fft2_r2c.hpp"
#include "ridgelet/rc_linearize.hpp"
#include "ridgelet/ridgelet_cell_array.hpp"
#include "ridgelet/ridgelet_frame.hpp"
#include "ridgelet/rt.hpp"
#include "spectral/quadrature/qhermitew.hpp"

typedef FFTr2c<PlannerR2C> fft_t;
// typedef FFTr2c<PlannerR2COD> fft_t;

unsigned int PLANNER_STRATEGY = FFTW_ESTIMATE;

typedef RT<double, RidgeletFrame, fft_t> RT_t;
typedef RT_t::rt_coeff_t rt_coeff_t;
typedef RidgeletCellArray<rt_coeff_t> rca_t;

namespace po = boost::program_options;

unsigned int ncoeffs(const RidgeletFrame& rf)
{
  unsigned int nc = 0;
  for (auto& lam : rf.lambdas()) {
    auto t = tgrid_dim(lam, rf);
    nc += std::get<0>(t) * std::get<1>(t);
  }
  return nc;
}

std::vector<std::tuple<int, int>> load_directions_all(int K)
{
  std::vector<std::tuple<int, int>> vec(K * K);
  for (int i = 0; i < K; ++i) {
    for (int j = 0; j < K; ++j) {
      vec[i * K + j] = std::make_tuple(i, j);
    }
  }

  return vec;
}

std::vector<std::tuple<int, int>> load_directions_file(const std::string& fname = "config.yaml")
{
  if (!(boost::filesystem::is_regular_file(fname) || boost::filesystem::is_symlink(fname))) {
    throw std::runtime_error("No file called " + fname + "found! Exiting.");
  }
  YAML::Node config = YAML::LoadFile(fname);
  auto V = config["dirs"];

  std::vector<std::tuple<int, int>> vec;
  for (int k = 0; k < V.size(); ++k) {
    int i = V[k][0].as<int>();
    int j = V[k][1].as<int>();
    vec.push_back(std::make_tuple(i, j));
  }

  return vec;
}

std::vector<double> load_qs(const std::string& fname = "config.yaml")
{
  if (!(boost::filesystem::is_regular_file(fname) || boost::filesystem::is_symlink(fname))) {
    throw std::runtime_error("No file called " + fname + "found! Exiting.");
  }
  YAML::Node config = YAML::LoadFile(fname);
  auto V = config["qs"];

  std::vector<double> vec;
  for (int k = 0; k < V.size(); ++k) {
    vec.push_back(V[k].as<double>());
  }

  return vec;
}

int get_frame(const std::string& fname)
{
  std::regex my_regex(".*solution_vector([0-9]*)");
  std::smatch match;
  bool found = std::regex_search(fname, match, my_regex);
  if (found) {
    return atoi(match[1].str().c_str());
  } else {
    throw std::runtime_error("could not find frame number");
  }
}

int main(int argc, char* argv[])
{
  std::cout << "SOURCE::INFO::GIT_BRANCHNAME " << GIT_BNAME << "\n";
  std::cout << "SOURCE::INFO::GIT_SHA1       " << GIT_SHA1 << "\n";

  double q;
  po::options_description options("options");
  options.add_options()
      ("help", "produce help message")
      ("save", "save set of active coefficients")
      ("alld", "compute rt in all directions")
      ("verbose,v", "vebose output")
      ("errors", "compute errors");

  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, options), vm);
    po::notify(vm);
  } catch (std::exception& e) {
    if (vm.count("help"))
      std::cout << options << "\n";
    else
      std::cout << e.what() << "\n";
    return 1;
  }
  if (vm.count("help")) {
    std::cout << options << "\n";
    return 0;
  }

  typedef Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> array_t;
  // read input files from stdin and check if they exist
  std::string input_line;
  std::vector<std::string> fnames;
  while (std::cin) {
    std::getline(std::cin, input_line);
    // check for empty (e.g. terminating line)
    std::regex my_regex("^([[:space:]]*|)$");
    std::smatch my_match;
    std::regex_search(input_line, my_match, my_regex);
    // cout << "match size: " << my_match.size() << "\n";
    if (my_match.size() > 0) {
      break;
    } else {
      if (boost::filesystem::is_regular_file(input_line) ||
          boost::filesystem::is_symlink(input_line))
        fnames.push_back(input_line);
      else
        throw std::runtime_error("File " + input_line + " does not exist. Exiting.");
    }
  };

  std::cout << "Found " << fnames.size() << " input files\n";
  std::vector<int> frame_numbers(fnames.size());
  std::transform(fnames.begin(), fnames.end(), frame_numbers.begin(), get_frame);

  // load files
  std::shared_ptr<RidgeletFrame> rf_ptr;
  std::shared_ptr<RCLinearize> rcl_ptr;
  typedef std::vector<std::tuple<int, int>> vdirections_t;
  std::shared_ptr<vdirections_t> directions_ptr;
  std::vector<double> qs;                              // ratio of active coefficients.  q in (0, 1]
  directions_ptr = std::make_shared<vdirections_t>();  // initialize empty
  Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> active_coeffs;

  for (auto& fname : fnames) {
    array_t coeffs;
    hid_t h5f = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    eigen2hdf::load(h5f, "C", coeffs);
    int N = coeffs.rows();
    int L = coeffs.cols();
    int K = std::sqrt(N);
    ASSERT(K * K == N);
    int Lx = std::sqrt(L);
    ASSERT(Lx * Lx == L);
    H5Fclose(h5f);

    if (!rf_ptr) {
      int J = std::log2(2 * Lx) - 2;
      rf_ptr = std::make_shared<RidgeletFrame>(J, J, 1, 1);
      rcl_ptr = std::make_shared<RCLinearize>(*rf_ptr);
      fft_t fft;
      init_fftw(fft, PLANNER_STRATEGY, *rf_ptr);

      if (vm.count("alld")) {
        // read directions from config
        *directions_ptr = load_directions_all(K);
      } else {
        // make all directions
        *directions_ptr = load_directions_file("config.yaml");
      }
      qs = load_qs("config.yaml");
      std::cout << "what is inside dirs"
                << "\n";
      for (auto v : *directions_ptr) {
        std::cout << std::get<0>(v) << "\t" << std::get<1>(v) << "\n";
      }

      active_coeffs.resize(qs.size(), (long int)ncoeffs);
      active_coeffs.setZero();
    }
    fft_t fft;
    RT_t rt(*rf_ptr);

    typedef RT_t::complex_array_t complex_array_t;

    // for later use
    boltzmann::QHermiteW quad(1.0, K);
    auto w = quad.wts();
    decltype(w) sqrtw(K);
    std::transform(w.begin(), w.end(), sqrtw.begin(), [](double x) { return std::sqrt(x); });

    // approximation error for reduced basis
    std::vector<double> errors(qs.size(), 0);

    std::cout << "___parallel region___"
              << "\n";

#pragma omp parallel default(none) shared(active_coeffs,  \
                                          coeffs,         \
                                          rt,             \
                                          rf_ptr,         \
                                          rcl_ptr,        \
                                          fft,            \
                                          L,              \
                                          Lx,             \
                                          K,              \
                                          directions_ptr, \
                                          qs,             \
                                          errors,         \
                                          sqrtw,          \
                                          vm)
    {
      std::printf("enter parallel region\n");
      complex_array_t Fhh(Lx, Lx);
      complex_array_t Fh(2 * Lx, 2 * Lx);
      std::printf("make rca_t\n");
// rca_t f_rc(*rf_ptr);

#pragma omp for schedule(static)
      for (int k = 0; k < directions_ptr->size(); k++) {
        // read F..
        std::printf("entering parallel for\n");
        int j1 = std::get<0>(directions_ptr->operator[](k));
        std::printf("j1=%d\n", j1);
        int j2 = std::get<1>(directions_ptr->operator[](k));
        std::printf("j2=%d", j2);
        std::printf("processing j1=%d, j2=%d", j1, j2);
        //   Eigen::Map<array_t> F(coeffs.data()+L*(j2*K+j1), Lx, Lx);
        //   fft.ft(Fhh, F, false);
        //   Fh.setZero();
        //   ftcut(Fh, Lx, Lx) = Fhh;
        //   hf_zero(Fh);
        //   rt.rt(f_rc.coeffs(), Fh);

        //   std::vector<double> rclin = rcl_ptr->linearize(f_rc.coeffs());
        //   std::vector<double> tres = rcl_ptr->get_threshold(std::vector<double>(rclin), qs);

        //   for (unsigned int qi = 0; qi < tres.size(); ++qi) {
        //     // compute active set
        //     // find locations where std::abs(rclin) > tre
        //     for (unsigned int i = 0; i < rclin.size(); ++i) {
        //       if(std::abs(rclin[i]) > tres[qi])
        //         // synchronization is not necessary
        //         active_coeffs(qi,i) = 1;
        //     }
        //     // compute approximation errors
        //     if(vm.count("errors")) {
        //       rca_t reduced = f_rc;
        //       rcl_ptr->threshold(f_rc, tres[qi]);
        //       // irt
        //       complex_array_t Fh2(2*Lx, 2*Lx);
        //       rt.irt(Fh2, reduced);
        //       complex_array_t Fh(Lx, Lx);
        //       Fh = ftcut(Fh2, Lx, Lx);
        //       hf_zero(Fh);
        //       array_t F2(Lx, Lx);
        //       fft.ift(F2, Fh);
        //       double h = 1.0/Lx;
        //       double loc_err = ((F-F2).cwiseAbs2()).sum()*h*h*sqrtw[j1]*sqrtw[j2];
        //       #pragma omp critical
        //       {
        //         errors[qi] += loc_err;
        //       }
        //     }
        //   }
      }
    }
    // if(vm.count("errors")) {
    //   std::cout << fname << "@errors " << "\t";
    //   for (int i = 0; i < errors.size(); ++i) {
    //     std::cout << std::scientific << std::sqrt(errors[i]) << "\t";
    //   }
    //   std::cout << "\n";
    // }
    // // stats
    // Eigen::ArrayXd nnz_ratio = active_coeffs.rowwise().sum().cast<double>();
    // nnz_ratio /= active_coeffs.cols();
    // std::cout << fname   << "@nnz_ratio" << "\t";
    // for (int i = 0; i < active_coeffs.rows(); ++i) {
    //   std::cout << nnz_ratio[i] << "\t";
    // }
    // std::cout << "\n";

  }  // end for files
  // if(vm.count("save")) {
  //   int frame_min = *std::min_element(frame_numbers.begin(), frame_numbers.end());
  //   int frame_max = *std::max_element(frame_numbers.begin(), frame_numbers.end());

  //   char buf[256];
  //   std::sprintf(buf, "active_coeffs_%06d-%06d.h5", frame_min, frame_max);
  //   hid_t h5f = H5Fcreate(buf, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  //   Eigen::Map<Eigen::VectorXi> ac_map(active_coeffs.data(), active_coeffs.size());
  //   eigen2hdf::save(h5f, "C", ac_map);
  //   H5Fclose(h5f);
  // }

  return 0;
}
