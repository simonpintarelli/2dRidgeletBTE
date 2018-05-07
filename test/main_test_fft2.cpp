#include <base/eigen2hdf.hpp>
#include <base/init.hpp>
#include <cstdio>
#include <fft/fft2.hpp>
#include <iostream>

using namespace std;

const char* fname = "test_fft2.h5";

int main(int argc, char* argv[])
{
  SOURCE_INFO();
  FFT fft;

  typedef FFT::array_t array_t;
  typedef FFT::complex_array_t complex_array_t;

  //  std::vector<int> N0 = {2, 4, 8};
  std::vector<int> N0 = {2, 32, 64, 320, 333, 501};
  // std::vector<int> N1 = {2, 5, 8};

  // hid_t group, file;
  // file = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  for (int n0 : N0) {
    // for(int n1 : N1 ) {
    int n1 = n0;
    cout << "fft of size " << n0 << " x " << n1 << "\n";
    // char group_name[255];
    // sprintf(group_name, "/G%d_%d", n0, n1);
    // group = H5Gcreate(file, group_name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    array_t X(n0, n1);
    X.setRandom();
    //      cout << "X:\n" << X << "\n";
    complex_array_t Y;
    fft.fft2(Y, X, false);
    // eigen2hdf::save(group, "Y", Y);
    array_t X2;
    fft.ifft2(X2, Y);

    // cout << "X2: \n" << X2 << "\n";
    cout << "DIFF"
         << "\n"
         << ((X - X2).abs()).maxCoeff() << endl;

    // eigen2hdf::save(group, "X", X);

    // eigen2hdf::save(group, "X2", X2);
    // H5Gclose (group);
  }

  for (int n0 : N0) {
    // for(int n1 : N1 ) {
    int n1 = n0;
    cout << "fft of size " << n0 << " x " << n1 << "\n";
    // char group_name[255];
    // sprintf(group_name, "/G%d_%d", n0, n1);
    // group = H5Gcreate(file, group_name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    complex_array_t X(n0, n1);
    X.setRandom();
    //      cout << "X:\n" << X << "\n";
    complex_array_t Y;
    fft.fft2(Y, X, false);
    // eigen2hdf::save(group, "Y", Y);
    complex_array_t X2;
    fft.ifft2(X2, Y);

    // cout << "X2: \n" << X2 << "\n";
    cout << "DIFF"
         << "\n"
         << ((X - X2).abs()).maxCoeff() << endl;

    // eigen2hdf::save(group, "X", X);

    // eigen2hdf::save(group, "X2", X2);
    // H5Gclose (group);
  }

  //  H5Fclose(file);

  return 0;
}
