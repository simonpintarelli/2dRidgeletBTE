#include <boost/program_options.hpp>
#include <cstdio>
#include <iostream>
// own includes ------------------------------------------------------------
#include <fft/fft2_r2c.hpp>
#include "base/eigen2hdf.hpp"
#include "base/init.hpp"
#include "base/timer.hpp"
#include "fft/planner.hpp"

using namespace std;

namespace po = boost::program_options;

// --------------------------------------------------------------------------------
void test_r2c(po::variables_map vm)
{
  RDTSCTimer timer;
  typedef PlannerR2C planner_t;
  typedef FFTr2c<planner_t> fft_t;
  typedef fft_t::array_t array_t;
  typedef fft_t::complex_array_t complex_array_t;

  const int nrep = vm["nrep"].as<int>();
  std::vector<int> N0 = {64, 128, 256, 512, 1024, 2048};

  fft_t fft;
  auto& planner = fft.get_plan();
  if (vm.count("measure")) {
    cout << "FFTW_MEASURE"
         << "\n";
    planner.set_flags(FFTW_MEASURE);
  } else {
    cout << "FFTW_ESTIMATE"
         << "\n";
    planner.set_flags(FFTW_ESTIMATE);
  }

  for (unsigned int i = 0; i < N0.size(); ++i) {
    planner.create_and_get_plan(N0[i], N0[i], planner_t::FWD, ft_type::R2C);
    planner.create_and_get_plan(N0[i], N0[i], planner_t::INV, ft_type::R2C);
  }

  for (unsigned int i = 0; i < N0.size(); ++i) {
    int n = N0[i];
    array_t X(n, n);
    X.setRandom();
    complex_array_t Y(n, n);
    timer.start();
    for (int jj = 0; jj < nrep; ++jj) {
      fft.ft(Y, X, false);
    }
    auto tlap = timer.stop();
    timer.start();
    array_t X2(n, n);
    for (int jj = 0; jj < nrep; ++jj) {
      fft.ift(X2, Y);
    }
    auto tlap_inv = timer.stop();

    timer.print(cout, tlap / nrep, "FFT size " + boost::lexical_cast<string>(n));
    timer.print(cout, tlap_inv / nrep, "IFFT size " + boost::lexical_cast<string>(n));
    auto Diff = X2 - X;
    cout << "diff: " << Diff.abs().sum() << "\n";
  }
}

// --------------------------------------------------------------------------------
void test_c2c(po::variables_map vm)
{
  RDTSCTimer timer;
  typedef PlannerR2C planner_t;
  typedef FFTr2c<planner_t> fft_t;
  typedef fft_t::complex_array_t complex_array_t;

  std::vector<int> N0 = {64, 128, 256, 512, 1024, 2048};
  const int nrep = vm["nrep"].as<int>();
  fft_t fft;
  auto& planner = fft.get_plan();
  if (vm.count("measure")) {
    cout << "FFTW_MEASURE"
         << "\n";
    planner.set_flags(FFTW_MEASURE);
  } else {
    cout << "FFTW_ESTIMATE"
         << "\n";
    planner.set_flags(FFTW_ESTIMATE);
  }

  for (unsigned int i = 0; i < N0.size(); ++i) {
    planner.create_and_get_plan(N0[i], N0[i], planner_t::FWD, ft_type::C2C);
    planner.create_and_get_plan(N0[i], N0[i], planner_t::INV, ft_type::C2C);
  }
  planner.print(cout);

  for (unsigned int i = 0; i < N0.size(); ++i) {
    int n = N0[i];
    complex_array_t X(n, n);
    X.setRandom();
    complex_array_t Y(n, n);
    timer.start();
    for (int jj = 0; jj < nrep; ++jj) {
      fft.ft(Y, X, false);
    }
    auto tlap = timer.stop();
    complex_array_t X2(n, n);
    timer.start();
    for (int jj = 0; jj < nrep; ++jj) {
      fft.ift(X2, Y);
    }
    auto tlap_inv = timer.stop();
    auto Diff = X2 - X;
    cout << "diff: " << Diff.abs().sum() << "\n";

    timer.print(cout, tlap / nrep, "FFT size " + boost::lexical_cast<string>(n));
    timer.print(cout, tlap_inv / nrep, "IFFT size " + boost::lexical_cast<string>(n));
  }
}

int main(int argc, char* argv[])
{
  SOURCE_INFO();
  po::options_description options("options");
  int nrep;
  options.add_options()
      ("help", "produce help message")
      ("measure", "use FFTW_MEASURE")
      ("nrep", po::value<int>(&nrep)->default_value(1), "num. repetitions");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, options), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << options << "\n";
    return 0;
  }

  cout << "nrep: " << nrep << "\n";

  cout << "==================== complex<->complex fft..."
       << "\n";
  test_c2c(vm);
  cout << "==================== real<->complex fft..."
       << "\n";
  test_r2c(vm);

  return 0;
}
