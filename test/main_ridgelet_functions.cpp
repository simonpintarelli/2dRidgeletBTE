#include <Eigen/Core>
#include <iostream>
#include <ridgelet/construction/ridgelet_functions.hpp>

using namespace std;

int main(int argc, char *argv[])
{
  unsigned int N = 100;
  Eigen::VectorXd vec(N);
  Eigen::VectorXd ty(N);

  vec.setLinSpaced(N, -2, 2);

  TransitionFunction t;

  std::transform(vec.data(), vec.data() + vec.size(), ty.data(), t);
  cout << "Transfer function:"
       << "\n";
  cout << "x: " << vec.array().transpose() << endl;
  cout << "y: " << ty.array().transpose() << endl;

  PsiSpherical1<> psi_spherical;

  Eigen::VectorXd ws(N);
  std::transform(vec.data(), vec.data() + vec.size(), ws.data(), psi_spherical);

  cout << "Spherical: "
       << "\n";
  cout << "y" << ws.array().transpose() << "\n";

  return 0;
}
