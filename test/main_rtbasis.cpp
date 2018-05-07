#include <boost/program_options.hpp>

#include "base/init.hpp"
#include "ridgelet/basis.hpp"
#include "ridgelet/lambda.hpp"
#include "ridgelet/ridgelet_frame.hpp"

using namespace std;

int main(int argc, char* argv[])
{
  int J, rho;

  SOURCE_INFO();
  namespace po = boost::program_options;

  po::options_description options("options");
  options.add_options()("help", "produce help message")
      ("Jxy,J", po::value<int>(&J)->default_value(3), "J")
      ("rho, r", po::value<int>(&rho)->default_value(1), "rho_x")
      ("no-dump", "dump coefficients to hdf5")
      ("list", "list basis and sizes");
  po::variables_map vm;

  try {
    po::store(po::parse_command_line(argc, argv, options), vm);
    po::notify(vm);
    if (vm.count("help")) {
      std::cout << options << "\n";
      return 0;
    }
  } catch (std::exception& e) {
    cout << e.what() << "\n";
  }

  RidgeletFrame frame(J, J, rho, rho);

  RTBasis<> rt_basis;
  auto& lambdas = frame.lambdas();
  for (auto& lam : lambdas) {
    unsigned int tx = translation_size(lam.t, lam.j, rho, 0);
    unsigned int ty = translation_size(lam.t, lam.j, rho, 1);

    unsigned int incx = std::max(tx / 4, 1u);
    unsigned int incy = std::max(ty / 4, 1u);
    cout << "incx: " << incx << "\n";
    cout << "incy: " << incy << "\n";

    for (unsigned int i = 0; i < tx; i += incx) {
      for (unsigned int j = 0; j < ty; j += incy) {
        rt_basis.insert(lam, i / double(tx), j / double(ty));
      }
    }
  }

  auto& elements = rt_basis.get_elements();

  cout << "basis size: " << rt_basis.size() << "\n";

  for (unsigned int i = 0; i < elements.size(); ++i) {
    auto index = rt_basis.get_index(elements[i]);
    BOOST_VERIFY(index == i);
  }

  return 0;
}
