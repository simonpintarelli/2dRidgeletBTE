#include <iomanip>
#include <iostream>

#include "matrices/tensor_entries.hpp"
#include "ridgelet/lambda.hpp"
#include "ridgelet/ridgelet_frame.hpp"

int main(int argc, char* argv[])
{
  int J = 7;
  RidgeletFrame rf(J, J, 1, 1);


  auto l1 = lambda_t(6, rt_type::X, 1);
  auto l2 = lambda_t(6, rt_type::X, 1);
  auto l3 = lambda_t(6, rt_type::X, -2);
  auto& ft1 = rf.get_sparse(l1);
  auto& ft2 = rf.get_sparse(l2);
  auto& ft3 = rf.get_sparse(l3);

  std::cout << "ft1: " << std::setw(12) << l1 << "  shape: " << ft1.rows() << " " << ft1.cols()
            << std::endl;
  std::cout << "ft2: " << std::setw(12) << l2 << "  shape: " << ft2.rows() << " " << ft2.cols()
            << std::endl;
  std::cout << "ft3: " << std::setw(12) << l3 << "  shape: " << ft3.rows() << " " << ft3.cols()
            << std::endl;
  std::cout << "\n\n\n";

  std::cout << "results from old implementation in (..)\n";
  std::cout << "------------------------------"
            << "\n";
  {
    std::cout << "double vsum = overlap3_simple(ft1, ft2, ft3):" << std::endl;
    double vsum = overlap3_simple(ft1, ft2, ft3);
    double vold = overlap3(ft1, ft2, ft3);
    std::cout << std::setprecision(7) << vsum << "\t(" + std::to_string(vold) + ")"
              << "\n";
  }
  std::cout << "------------------------------"
            << "\n";
  {
    std::cout << "double vsum = overlap3_simple(ft2, ft3, ft1):" << std::endl;
    double vsum = overlap3_simple(ft2, ft3, ft1);
    double vold = overlap3(ft2, ft3, ft1);
    std::cout << std::setprecision(7) << vsum << "\t(" + std::to_string(vold) + ")"
              << "\n";
  }
  std::cout << "------------------------------"
            << "\n";
  {
    std::cout << "double vsum = overlap3_simple(ft3, ft1, ft2):" << std::endl;
    double vsum = overlap3_simple(ft3, ft1, ft2);
    double vold = overlap3(ft3, ft1, ft2);
    std::cout << std::setprecision(7) << vsum << "\t(" + std::to_string(vold) + ")"
              << "\n";
  }
  std::cout << "------------------------------"
            << "\n";
  {
    std::cout << "double vsum = overlap3_simple(ft3, ft2, ft1):" << std::endl;
    double vsum = overlap3_simple(ft3, ft2, ft1);
    double vold = overlap3(ft3, ft2, ft1);
    std::cout << std::setprecision(7) << vsum << "\t(" + std::to_string(vold) + ")"
              << "\n";
  }

  return 0;
}
