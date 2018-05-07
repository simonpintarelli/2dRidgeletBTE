#include <Eigen/Dense>
#include <iostream>

#include <base/init.hpp>
#include "fft/fft2.hpp"
#include "ridgelet/fold.hpp"

using namespace std;

typedef Eigen::ArrayXXd array_t;
typedef Eigen::ArrayXXcd complex_array_t;

int main(int argc, char *argv[])
{
  Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(24, 0, 1);

  Eigen::Map<const array_t> xm(x.data(), 4, 6);
  cout << "input"
       << "\n";
  cout << xm << "\n";
  array_t X = xm;

  array_t Y;
  fold(Y, X, 3, 1);

  cout << "fold(X, width=3, dim=1)"
       << "\n";
  cout << "Y:\n" << Y << "\n";

  array_t Y2;
  fold(Y2, X, 2, 0);

  cout << "fold(X, width=2, dim=0)"
       << "\n";
  cout << "Y:\n" << Y2 << "\n";

  return 0;
}
