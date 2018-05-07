// system includes -----------------------------------------------
#include <boost/math/constants/constants.hpp>
#include <boost/program_options.hpp>
#include <iostream>
// own includes --------------------------------------------------
#include <base/eigen2hdf.hpp>
#include <base/init.hpp>
#include <fft/fft2.hpp>
#include <fft/ft_grid_helpers.hpp>
#include <fft/shift.hpp>
#include <ridgelet/ridgelet_frame.hpp>
#include <ridgelet/rt.hpp>

using namespace std;

Eigen::VectorXd fft_freq(int n)
{
  Eigen::VectorXd x(n);

  for (int i = 0; i < n / 2 + n % 2; ++i) {
    x[i] = i;
  }
  for (int i = n / 2 + n % 2, j = 0; i < n; ++i, ++j) {
    x[i] = -n / 2 + j;
  }

  return x;
}

/// even/odd test
void test1()
{
  Eigen::VectorXd x5(5);
  Eigen::VectorXd x4(4);

  fftshift(x5, fft_freq(5), 0);
  fftshift(x4, fft_freq(4), 0);

  cout << "x5:\n" << x5.transpose() << "\n";
  cout << "x4:\n" << x4.transpose() << "\n";

  cout << "ftcut(x5, 4, 1):"
       << "\n";
  cout << ftcut(x5, 4, 1).transpose() << endl;

  // cout << "fft_freq(4):\n"  << fft_freq(4) << "\n";
  // cout << "fft_freq(5):\n"  << fft_freq(5) << "\n";
}

/// n vs n/2 test
void test2()
{
  Eigen::VectorXd x8(8);
  Eigen::VectorXd x4(4);

  fftshift(x8, fft_freq(8), 0);
  fftshift(x4, fft_freq(4), 0);

  cout << "x8:\n" << x8.transpose() << "\n";
  cout << "x4:\n" << x4.transpose() << "\n";

  cout << "ftcut(x5, 4, 1):"
       << "\n";
  cout << ftcut(x8, 4, 1).transpose() << endl;
}

int main(int argc, char *argv[])
{
  cout << "---------- test1() ----------"
       << "\n";
  test1();

  cout << "---------- test2() ----------"
       << "\n";
  test2();

  return 0;
}
