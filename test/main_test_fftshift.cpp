#include <Eigen/Dense>
#include <iostream>

#include <base/init.hpp>
#include "fft/shift.hpp"

using namespace std;

typedef Eigen::ArrayXXd array_t;

typedef Eigen::ArrayXd array1_t;

void test_rowwise()
{
  int n = 6;
  array1_t x = array1_t::LinSpaced(n, -n / 2, n / 2 - (n + 1) % 2);
  //  cout << x << "\n";
  array_t X(n, 10);
  X.colwise() = x;
  // --------------------------------------------------
  // test rowwise shift
  array1_t y = array1_t::LinSpaced(n, -n / 2, n / 2 - (n + 1) % 2);

  array_t Y(10, n);
  Y.rowwise() = y.transpose();

  cout << "---------- Y ----------"
       << "\n";
  cout << Y << "\n";

  cout << "---------- ifftshift(Y) ----------"
       << "\n";
  array_t iY = array_t::Zero(Y.rows(), Y.cols());
  ifftshift(iY, Y, 1);
  cout << iY << "\n\n";

  cout << "---------- fftshift(ifftshift(Y))  ----------"
       << "\n";
  array_t Y2 = array_t::Zero(Y.rows(), Y.cols());
  fftshift(Y2, iY, 1);
  cout << Y2 << "\n\n";
}

void test_colwise()
{
  int n = 6;
  array1_t x = array1_t::LinSpaced(n, -n / 2, n / 2 - (n + 1) % 2);
  cout << x << "\n";
  array_t X(n, 10);
  X.colwise() = x;
  // --------------------------------------------------
  // test colwise shift
  cout << "---------- X ----------"
       << "\n";
  cout << X << "\n";

  cout << "---------- ifftshift(X) ----------"
       << "\n";
  array_t iX = array_t::Zero(X.rows(), X.cols());
  ifftshift(iX, X, 0);
  cout << iX << "\n\n";

  cout << "---------- fftshift(ifftshift(X))  ----------"
       << "\n";
  array_t X2 = array_t::Zero(X.rows(), X.cols());
  fftshift(X2, iX, 0);
  cout << X2 << "\n\n";
}

void test2d()
{
  // test shift in both directions
  int nx = 6;
  int ny = 4;

  array_t XX(ny, nx);
  XX.setZero();
  int posx_max = nx / 2;
  int posy_max = ny / 2;

  XX.block(0, 0, posy_max, posx_max) = array_t::Ones(posy_max, posx_max);
  XX.block(posy_max, 0, ny - posy_max, nx) = 2 * array_t::Ones(ny - posy_max, nx);
  XX.block(0, posx_max, posy_max, nx - posx_max) = 3 * array_t::Ones(posy_max, nx - posx_max);
  XX.block(posy_max, posx_max, ny - posy_max, nx - posx_max) =
      4 * array_t::Ones(ny - posy_max, nx - posx_max);

  cout << "---------- XX ----------"
       << "\n";
  cout << XX << endl;

  cout << "---------- fftshift(XX) ----------"
       << "\n";
  array_t YY(ny, nx);
  YY.setZero();
  fftshift(YY, XX);
  cout << YY << "\n";

  cout << "---------- ifftshift(fftshift(XX)) ----------"
       << "\n";
  array_t XX2(ny, nx);
  XX2.setZero();
  ifftshift(XX2, YY);
  cout << XX2 << "\n";
}

void test2d_v2()
{
  // test shift in both directions
  int nx = 6;
  int ny = 4;

  array_t XX(ny, nx);
  array_t YY(ny, nx);
  XX.setZero();
  YY.setZero();
  int posx_max = nx / 2;
  int posy_max = ny / 2;

  XX = Eigen::ArrayXd::LinSpaced(nx, -posx_max, posx_max - 1).transpose().replicate(ny, 1);
  YY = Eigen::ArrayXd::LinSpaced(ny, -posy_max, posy_max - 1).replicate(1, nx);

  cout << "---------- ifftshift(fftshift(XX)) ----------"
       << "\n";
  array_t XXt(ny, nx);
  XXt.setZero();
  fftshift(XXt, XX);
  array_t XX2(ny, nx);
  ifftshift(XX2, XXt);
  {
    double diff = (XX2 - XX).abs().sum();
    cout << "diff= " << diff << endl;
  }

  cout << "---------- ifftshift(fftshift(YY)) ----------"
       << "\n";
  array_t YYt(ny, nx);
  YYt.setZero();
  fftshift(YYt, YY);
  array_t YY2(ny, nx);
  ifftshift(YY2, YYt);
  {
    double diff = (YY2 - YY).abs().sum();
    cout << "diff= " << diff << endl;
  }
}

int main(int argc, char *argv[])
{
  SOURCE_INFO();

  cout << "---------- test rowwise ----------"
       << "\n";
  test_rowwise();

  cout << "---------- test colwise ----------"
       << "\n";
  test_colwise();

  cout << "---------- test 2d ----------"
       << "\n";
  test2d_v2();

  return 0;
}
