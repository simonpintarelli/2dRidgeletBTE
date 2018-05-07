// Compute and store ridgelet coefficients in linearized form
// Input: fsolution*h5 file(s) computed with bte_omp_ftcg executable

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
#include <stdexcept>
#include <string>
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

unsigned int PLANNER_STRATEGY = FFTW_MEASURE;

typedef RT<double, RidgeletFrame, fft_t> RT_t;
typedef RT_t::rt_coeff_t rt_coeff_t;
typedef RidgeletCellArray<rt_coeff_t> rca_t;

namespace po = boost::program_options;

std::string out_fname = "rt_coeffs.h5";

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

unsigned int get_ncoeffs(const RidgeletFrame& rf)
{
  unsigned int nc = 0;
  for (auto& lam : rf.lambdas()) {
    auto t = tgrid_dim(lam, rf);
    unsigned int TX = std::get<0>(t) * std::get<1>(t);
    nc += TX;
  }
  return nc;
}


bool is_power_of_two(int x)
{
  return (x>0) && !(x & (x-1));
}

int main(int argc, char* argv[])
{
  std::cout << "SOURCE::INFO::GIT_BRANCHNAME " << GIT_BNAME << "\n";
  std::cout << "SOURCE::INFO::GIT_SHA1       " << GIT_SHA1 << "\n";

  // disable buffering of printf
  std::setbuf(stdout, NULL);

  po::options_description options("options");
  options.add_options()
      ("help", "produce help message")
      ("dst", po::value<std::string>(), "write output to dst");

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

  if (vm.count("dst")) {
    out_fname = vm["dst"].as<std::string>();
  }
  std::cerr << "writing output to " << out_fname << std::endl;

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

  // load files
  std::shared_ptr<RidgeletFrame> rf_ptr;
  std::shared_ptr<RCLinearize> rcl_ptr;

  hid_t h5f_out = H5Fcreate(out_fname.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  for (auto& fname : fnames) {
    int frame = get_frame(fname);
    array_t coeffs;
    hid_t h5f = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    eigen2hdf::load(h5f, "C", coeffs);
    int N = coeffs.rows();
    int L = coeffs.cols();
    int K = std::sqrt(N);
    ASSERT(K * K == N);
    int Lx = std::sqrt(L);
    std::cout << "Lx: " << Lx << "\n";
    ASSERT(Lx * Lx == L);
    ASSERT(is_power_of_two(Lx));
    H5Fclose(h5f);

    if (!rf_ptr) {
      int J = std::log2(2 * Lx) - 2;
      std::cout << "J: "  << J << "\n";
      rf_ptr = std::make_shared<RidgeletFrame>(J, J, 1, 1);
      rcl_ptr = std::make_shared<RCLinearize>(*rf_ptr);
      fft_t fft;
      init_fftw(fft, PLANNER_STRATEGY, *rf_ptr);
    }
    fft_t fft;
    RT_t rt(*rf_ptr);

    typedef RT_t::complex_array_t complex_array_t;

    // store Ridgelet coefficients for each direction in the following array:
    typedef Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> out_array_t;

    int ncoeffs = get_ncoeffs(*rf_ptr);
    out_array_t RTCOEFFS(K * K, ncoeffs);

#pragma omp parallel
    {
      complex_array_t Fhh(Lx, Lx);
      complex_array_t Fh(2 * Lx, 2 * Lx);
      rca_t f_rc(*rf_ptr);

#pragma omp for schedule(dynamic) collapse(2)
      for (int j1 = 0; j1 < K; j1++) {
        for (int j2 = 0; j2 < K; j2++) {
          // read F..
          int idv = j2 * K + j1;
          Eigen::Map<array_t> F(coeffs.data() + L * idv, Lx, Lx);
          fft.ft(Fhh, F, false);
          Fh.setZero();
          ftcut(Fh, Lx, Lx) = Fhh;
          hf_zero(Fh);
          rt.rt(f_rc.coeffs(), Fh);

          std::vector<double> rclin = rcl_ptr->linearize(f_rc.coeffs());

          // valgrind complains about the following
          // std::copy(rclin.begin(), rclin.end(), RTCOEFFS.row(idv).data());
          Eigen::Map<Eigen::VectorXd> vrclin(rclin.data(), rclin.size());
          RTCOEFFS.row(idv) = vrclin;
        }  // end for j2
      }    // end for j1
    }      // end omp parallel

    // save to hdf
    eigen2hdf::save(h5f_out, std::to_string(frame), RTCOEFFS);

  }  // end for files
  H5Fclose(h5f_out);

  return 0;
}
