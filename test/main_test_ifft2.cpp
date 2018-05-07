#include <Eigen/Dense>
#include <complex>
#include <iostream>

#include <fft/fft2.hpp>

using namespace std;

int main(int argc, char *argv[])
{
  typedef std::complex<double> cdouble;

  typedef Eigen::Array<cdouble, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> complex_array_t;
  typedef Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> array_t;

  complex_array_t f_tilde_hat(4, 4);

  f_tilde_hat << 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1;
  array_t f;

  FFT fft;

  std::cout << "f_tilde_hat:\n" << f_tilde_hat << "\n";

  fft.ifft2(f, f_tilde_hat);
  std::cout << "f:\n" << f << "\n";

  return 0;
}
