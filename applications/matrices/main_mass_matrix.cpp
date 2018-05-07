// restored from rdiff-backup: 2016-09-19T17:24:46+02:00

#include <omp.h>
#include <Eigen/Sparse>
#include <algorithm>
#include <boost/math/constants/constants.hpp>
#include <boost/program_options.hpp>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>

#include <base/eigen2hdf.hpp>
#include <fft/ft_grid_helpers.hpp>
#include <matrices/matrix_entries.hpp>
#include <ridgelet/construction/ft.hpp>
#include <ridgelet/construction/translation_grid.hpp>
#include <ridgelet/lambda.hpp>
#include <ridgelet/ridgelet_frame.hpp>
#include "base/init.hpp"

using namespace std;

const double Lx = 1.0;
const double Ly = 1.0;
const double tol = 1e-10;

const double PI = boost::math::constants::pi<double>();

int main(int argc, char* argv[])
{
  int J, rho;
  SOURCE_INFO();

  namespace po = boost::program_options;

  po::options_description options("options");
  options.add_options()("help", "produce help message")
      ("Jxy,J", po::value<int>(&J)->default_value(3), "J")
      ("rho, r", po::value<int>(&rho)->default_value(1), "rho_x")
      ("transport", "compute transport part too");
  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, options), vm);
    po::notify(vm);
    if (vm.count("help")) {
      std::cout << options << "\n";
      return 0;
    }
  } catch (std::exception& e) {
    std::cout << e.what() << "\n";
  }

  std::cout << "J:   " << J << "\n"
            << "rho: " << rho << "\n";

  RidgeletFrame frame(J, J, rho, rho);

  std::string fname = std::string("mass_matrix_entries") + std::to_string(J) + ".dat";
  // compute the translation grid sizes for reach lambda
  // and
  cout << "Translation grid..."
       << "\n";
  std::vector<unsigned int> tgrid_size(frame.size());
  std::vector<unsigned int> num_elements(frame.size() + 1, 0);
  auto& lambdas = frame.lambdas();
  for (unsigned int i = 0; i < frame.size(); ++i) {
    auto T = tgrid_dim(lambdas[i], frame);
    tgrid_size[i] = std::get<0>(T) * std::get<1>(T);
  }
  std::partial_sum(tgrid_size.begin(), tgrid_size.end(), num_elements.begin() + 1);
  {
    std::ofstream fout(std::string("num_elems") + std::to_string(J) + ".dat");
    std::for_each(
        num_elements.begin(), num_elements.end(), [&fout](unsigned int i) { fout << i << "\n"; });
    fout.close();
  }

  // Normalization ------------------------------
  cout << "Normalization..."
       << "\n";
  std::vector<double> ft_coeff_norms(frame.size());
#pragma omp parallel for
  for (unsigned int i = 0; i < frame.size(); ++i) {
    auto lambda = frame.lambdas()[i];
    if (lambda.t == rt_type::S) {
      ft_coeff_norms[i] = std::sqrt(frame.get_dense(lambda).cwiseAbs2().sum());
    } else {
      ft_coeff_norms[i] = std::sqrt(frame.get_sparse(lambda).cwiseAbs2().sum());
    }
  }

  /*
   *  key = (l_i, l_j, t_d)
   *  l_i, l_j, integer:
   *  ------------------
   *
   *  hint: lambda_i = lambdas[l_i]
   *
   *  difference translation grid t_d:
   *  -----------------
   *  tuple t_d = {tx_i - tx_j, ty_i - ty_j}
   */
  typedef std::tuple<int, int> tgrid_diff_key; /*  type of t_d */
  typedef std::tuple<int, int, tgrid_diff_key> key_type;
  std::unordered_map<key_type, double> mass_matrix_entries;
  std::unordered_map<key_type, double> transport_matrix_entries;

  cout << "Computing integrals..."
       << "\n";
#pragma omp parallel for schedule(dynamic, 1)
  for (unsigned int i = 0; i < frame.size(); ++i) {
    for (unsigned int j = 0; j <= i; ++j) {
      // skip pairs that are known to evaluate to zero
      if (std::abs(lambdas[i].j - lambdas[j].j) > 1) continue;
      if ((lambdas[i].t == rt_type::X && lambdas[j].t == rt_type::Y) ||
          (lambdas[i].t == rt_type::Y && lambdas[j].t == rt_type::X))
        continue;
      if ((lambdas[i].t == lambdas[j].t) && (lambdas[i].j == lambdas[j].j) &&
          (std::abs(lambdas[i].k - lambdas[j].k) > 1))
        continue;

      auto& Psih_i = frame.get_sparse(lambdas[i]);
      // Psih_i, Psih_j sparse
      auto& Psih_j = frame.get_sparse(lambdas[j]);

      auto T1 = tgrid_dim(lambdas[i], frame);
      auto T2 = tgrid_dim(lambdas[j], frame);

      int tx = std::max(std::get<0>(T1), std::get<0>(T2));
      int ty = std::max(std::get<1>(T1), std::get<1>(T2));

      // translation grid
      auto vtx = Eigen::ArrayXd::LinSpaced(tx, 0, (1 - 1 / double(tx)) * Lx);
      auto vty = Eigen::ArrayXd::LinSpaced(ty, 0, (1 - 1 / double(ty)) * Ly);

      double fnorm = 1 / (ft_coeff_norms[i] * ft_coeff_norms[j]);
      for (int idx_tx = 0; idx_tx < tx; ++idx_tx) {
        for (int idx_ty = 0; idx_ty < ty; ++idx_ty) {
          auto k = std::make_tuple(idx_tx, idx_ty);
          auto key = std::make_tuple(i, j, k);
          double v = compute_mass_entry(Psih_i, Psih_j, vtx[idx_tx], vty[idx_ty]);
          v *= fnorm;
          if (std::abs(v) > tol) {
#pragma omp critical
            mass_matrix_entries[key] = v;
          }

          if (vm.count("transport")) {
            double vxy = compute_tentry(
                Psih_i, Psih_j, vtx[idx_tx], vty[idx_ty], derivative_t::dX, derivative_t::dY);
            vxy *= fnorm;
            double vxx = compute_tentry(
                Psih_i, Psih_j, vtx[idx_tx], vty[idx_ty], derivative_t::dX, derivative_t::dX);
            vxx *= fnorm;
            double vyy = compute_tentry(
                Psih_i, Psih_j, vtx[idx_tx], vty[idx_ty], derivative_t::dY, derivative_t::dY);
            vyy *= fnorm;

            double vt = 2 * vxy + vxx + vyy;

            if (std::abs(vt) > tol) {
#pragma omp critical
              transport_matrix_entries[key] = vt;
            }
          }
        }
      }
    }  // end for inner lambda
  }    // end for outer lambda

  std::cout << "mass_matrix_entries.size: " << mass_matrix_entries.size() << "\n";
  std::cout << "writing results to " << fname << "\n";
  // write to disk
  std::ofstream fout(fname);
  for (auto& elem : mass_matrix_entries) {
    auto& key = elem.first;
    auto val = elem.second;
    int i = std::get<0>(key);
    int j = std::get<1>(key);
    int tx, ty;
    std::tie(tx, ty) = std::get<2>(key);
    fout << i << "\t" << j << "\t" << tx << "\t" << ty << "\t" << val << "\n";
  }
  fout.close();
  cout << "\tdone"
       << "\n";

  if (vm.count("transport")) {
    std::string fname = std::string("transport_matrix_" + std::to_string(J) + ".dat");
    std::ofstream fout(fname);
    for (auto& elem : transport_matrix_entries) {
      auto& key = elem.first;
      auto val = elem.second;
      int i = std::get<0>(key);
      int j = std::get<1>(key);
      int tx, ty;
      std::tie(tx, ty) = std::get<2>(key);
      fout << i << "\t" << j << "\t" << tx << "\t" << ty << "\t" << val << "\n";
    }
    fout.close();
  }

  return 0;
}
