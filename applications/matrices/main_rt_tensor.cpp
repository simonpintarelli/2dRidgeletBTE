#include <hdf5.h>
#include <omp.h>
#include <boost/assert.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include "base/eigen2hdf.hpp"
#include "base/init.hpp"
#include "matrices/tensor_entries.hpp"
#include "ridgelet/basis.hpp"
#include "ridgelet/rc_linearize.hpp"

struct tensor_entry_struct_t
{
  tensor_entry_struct_t(int i1_, int i2_, int i3_, double v_)
      : i1(i1_)
      , i2(i2_)
      , i3(i3_)
      , v(v_)
  { /* empty */
  }

  int i1;
  int i2;
  int i3;
  double v;
};

struct index_struct_t
{
  index_struct_t(int i1_, int i2_, int i3_)
      : i1(i1_)
      , i2(i2_)
      , i3(i3_)
  { /* empty */
  }

  int i1;
  int i2;
  int i3;
};

typedef std::vector<tensor_entry_struct_t> tensor_entries_t;

RTBasis<> make_basis_from_acfile(const RidgeletFrame& rf, const std::string& fname)
{
  if (!boost::filesystem::exists(fname)) {
    throw std::runtime_error("File " + fname + " not found!");
  }

  std::ifstream ifile;
  ifile.open(fname);
  std::string line;

  RCLinearize rcl(rf);

  RTBasis<> rt_basis;

  int lc = 0;  // line count
  while (std::getline(ifile, line)) {
    int fl = atoi(line.c_str());
    if (fl == 1) {
      lambda_t ll;
      std::tuple<double, double> tv;
      std::tie(ll, tv) = rcl.get_lambda_tt(lc);
      rt_basis.insert(ll, std::get<0>(tv), std::get<1>(tv));
      lc++;
    } else if (fl == 0) {
      lc++;
      continue;
    } else {
      throw std::runtime_error("invalid entry in " + fname);
    }
  }

  if (!(lc == rcl.size())) {
    throw std::runtime_error("size mismatch, expected " + std::to_string(rcl.size()) + " got " +
                             std::to_string(lc));
  }

  std::cout << "done parsing " << fname << "\n";
  std::cout << "rt_basis.size: " << rt_basis.size() << "\n";

  return rt_basis;
}

void tensor2hdf(const std::string& filename, const tensor_entries_t& entries)
{
  hid_t file = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  hid_t filetype, memtype, space, dset;
  /* Handles */
  herr_t status;
  hsize_t dims[1] = {entries.size()};
  typedef typename tensor_entries_t::value_type value_t;

  memtype = H5Tcreate(H5T_COMPOUND, sizeof(value_t));
  status = H5Tinsert(memtype, "i1", HOFFSET(value_t, i1), H5T_STD_I32LE);
  status = H5Tinsert(memtype, "i2", HOFFSET(value_t, i2), H5T_STD_I32LE);
  status = H5Tinsert(memtype, "i3", HOFFSET(value_t, i3), H5T_STD_I32LE);
  status = H5Tinsert(memtype, "v", HOFFSET(value_t, v), H5T_NATIVE_DOUBLE);

  filetype = H5Tcreate(H5T_COMPOUND, 3 * 4 + 8);
  status = H5Tinsert(filetype, "i1", 0, H5T_STD_I32LE);
  status = H5Tinsert(filetype, "i2", 4, H5T_STD_I32LE);
  status = H5Tinsert(filetype, "i3", 4 + 4, H5T_STD_I32LE);
  status = H5Tinsert(filetype, "v", 4 + 4 + 4, H5T_IEEE_F64LE);
  /*
   * Create dataspace.  Setting maximum size to NULL sets the maximum
   * size to be the current size.
   */
  space = H5Screate_simple(1, dims, NULL);

  /*
   * Create the dataset and write the compound data to it.
   */
  dset = H5Dcreate(file, "values", filetype, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Dwrite(dset, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, entries.data());
  if (status < 0) throw std::runtime_error("failed to write entries.data");

  /*
   * Close and release resources.
   */
  status = H5Dclose(dset);
  status = H5Sclose(space);
  status = H5Tclose(filetype);
  H5Fclose(file);
}

void entries_multiplicity2hdf(const std::string& filename,
                              const std::vector<index_struct_t>& indices,
                              const std::vector<unsigned long>& offsets)
{
  hid_t file = H5Fopen(filename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
  if (file < 0) throw std::runtime_error("cannot open file " + filename);
  hid_t filetype, memtype, space, dset;
  /* Handles */
  herr_t status;
  hsize_t dims[1] = {indices.size()};

  memtype = H5Tcreate(H5T_COMPOUND, sizeof(index_struct_t));
  status = H5Tinsert(memtype, "i1", HOFFSET(index_struct_t, i1), H5T_STD_I32LE);
  status = H5Tinsert(memtype, "i2", HOFFSET(index_struct_t, i2), H5T_STD_I32LE);
  status = H5Tinsert(memtype, "i3", HOFFSET(index_struct_t, i3), H5T_STD_I32LE);

  filetype = H5Tcreate(H5T_COMPOUND, 3 * 4);
  status = H5Tinsert(filetype, "i1", 0, H5T_STD_I32LE);
  status = H5Tinsert(filetype, "i2", 4, H5T_STD_I32LE);
  status = H5Tinsert(filetype, "i3", 4 + 4, H5T_STD_I32LE);
  /*
   * Create dataspace.  Setting maximum size to NULL sets the maximum
   * size to be the current size.
   */
  space = H5Screate_simple(1, dims, NULL);

  /*
   * Create the dataset and write the compound data to it.
   */
  dset = H5Dcreate(file, "indices", filetype, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Dwrite(dset, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, indices.data());

  /*
   * Close and release resources.
   */
  status = H5Dclose(dset);
  status = H5Sclose(space);
  status = H5Tclose(filetype);

  /* write offsets to file */
  dims[0] = offsets.size();
  space = H5Screate_simple(1, dims, NULL);
  dset = H5Dcreate(file, "offsets", H5T_NATIVE_ULONG, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Dwrite(dset, H5T_NATIVE_ULONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, offsets.data());
  if (status < 0) throw std::runtime_error("failed to write offsets.data");
  status = H5Dclose(dset);
  status = H5Sclose(space);

  H5Fclose(file);
}

int main(int argc, char* argv[])
{
  std::cout << sizeof(tensor_entry_struct_t) << "\n";
  std::cout << sizeof(index_struct_t) << "\n";
  unsigned int Jx;
  double tol;
  namespace po = boost::program_options;
  po::options_description options("options");
  options.add_options()("help", "produce help message")
      ("JJ,J", po::value<unsigned int>(&Jx)->required(), "Jx")
      ("in", po::value<std::string>()->required(),
      "boolen vector (as txt), containing active coefficients")
      ("out", po::value<std::string>()->default_value("rt_tensor.h5"), "hdf5 output file")
      ("tol", po::value<double>(&tol)->default_value(1e-12), "drop entries below tol")
      ("verbose", "verbose output");

  SOURCE_INFO();
  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, options), vm);
    po::notify(vm);
  } catch (std::exception& e) {
    std::cout << options << "\n";
    return 1;
  }
  if (vm.count("help")) {
    std::cout << options << "\n";
    return 0;
  }

  std::cout << "Drop entries tol: " << std::scientific << tol << "\n";
  std::cout << "J: " << Jx - 1 << "\n";  // create rt frame of this size (see below)

  // -1 like in bte_omp_ftcg
  RidgeletFrame rf(Jx - 1, Jx - 1, 1, 1);

  RTBasis<> rt_basis = make_basis_from_acfile(rf, vm["in"].as<std::string>());
  rt_basis.sort();
  std::cout << "Write (sorted) active basis elements to `rt_basis.desc`"
            << "\n";
  rt_basis.write_desc("rt_basis.desc");

  // collect lambda appearing in active set (e.g. rt_basis)
  auto lambdas = rf.lambdas();
  std::sort(lambdas.begin(), lambdas.end());
  // lambdas contained in rt_basis
  std::vector<lambda_t> lambdas_ac;
  for (auto& ll : lambdas) {
    auto beg = rt_basis.get_beg(ll);
    auto end = rt_basis.get_end(ll);
    int size = end - beg;
    if (size > 0) lambdas_ac.push_back(ll);
    if (vm.count("verbose")) std::cout << ll << ", size: " << end - beg << "\n";
  }

  long N = rt_basis.size();
  // FUZZY value for mapping tgrid vector to integer
  long FUZZY = 2l << 30;
  // upper-bound for number of entries
  long nentries_ub = (N + 1) * (N + 2) * (N + 3) / 6;
  std::cout << "Symmetric tensor has: " << nentries_ub << " entries.\n";

  tensor_entries_t tensor_entries;
  // this might become huge!
  std::vector<index_struct_t> entries_multiplicity;
  std::vector<unsigned long> entries_multiplicity_offsets;
  unsigned long em_offset_ctr = 0;

  int lN = lambdas_ac.size();

#pragma omp parallel
#pragma omp single nowait
  for (int lindex1 = 0; lindex1 < lN; ++lindex1) {
    for (int lindex2 = 0; lindex2 <= lindex1; ++lindex2) {
      for (int lindex3 = 0; lindex3 <= lindex2; ++lindex3) {
        // loop over tgrid and collect unique (t12, t23)
        auto l1 = lambdas_ac[lindex1];
        auto l2 = lambdas_ac[lindex2];
        auto l3 = lambdas_ac[lindex3];
        typedef std::tuple<long, long> tdiff_t;
        typedef std::tuple<tdiff_t, tdiff_t> key_t;
        typedef std::tuple<long, long> tdiff_t;

        std::unordered_map<key_t, std::vector<index_struct_t>> entries;

        // On diagonal blocks, we do not iterate over the full translation grid.
        // See loop-limits for it2 and it3 below.
        bool is_diag = (l1 == l2 && l2 == l3) ? true : false;
#pragma omp taskgroup
        for (auto it1 = rt_basis.get_beg(l1); it1 < rt_basis.get_end(l1); ++it1) {
          for (auto it2 = rt_basis.get_beg(l2);
               is_diag ? (it2 <= it1) : (it2 < rt_basis.get_end(l2));
               ++it2) {
            for (auto it3 = rt_basis.get_beg(l3);
                 is_diag ? (it3 <= it2) : (it3 < rt_basis.get_end(l3));
                 ++it3) {
              int i1 = it1.idx();
              int i2 = it2.idx();
              int i3 = it3.idx();

              // compute only lower-diagonal entries of the tensor
              if (!(i1 >= i2 && i2 >= i3)) continue;
              index_struct_t tensor_index(i1, i2, i3);

              // create integer valued tgrid difference vectors (for checking if the value has
              // already been computed)
              tdiff_t t12k(FUZZY * (it1->get_iy() - it2->get_iy()),
                           FUZZY * (it1->get_ix() - it2->get_ix()));
              tdiff_t t23k(FUZZY * (it2->get_iy() - it3->get_iy()),
                           FUZZY * (it2->get_ix() - it3->get_ix()));

              // get translation grid difference vectors
              Eigen::Vector2d t12v = {it1->get_iy() - it2->get_iy(), it1->get_ix() - it2->get_ix()};
              Eigen::Vector2d t23v = {it2->get_iy() - it3->get_iy(), it2->get_ix() - it3->get_ix()};
              key_t key(t12k, t23k);

              auto it = entries.find(key);
              if (it != entries.end()) {
                // found
                it->second.push_back(tensor_index);
              } else {
                // compute and add to tensor
                entries.insert(std::make_pair(key, std::vector<index_struct_t>{tensor_index}));
#pragma omp task default(none) shared(tensor_entries, rf, tol) \
                                          firstprivate(l1, l2, l3, t12v, t23v, tensor_index)
                {
                  const auto& ft1 = rf.get_sparse(l1);
                  const auto& ft2 = rf.get_sparse(l2);
                  const auto& ft3 = rf.get_sparse(l3);
                  double norm_factor = rf.rf_norm(l1) * rf.rf_norm(l2) * rf.rf_norm(l3);
                  double val = overlap3_simple(ft1, ft2, ft3, t12v, t23v) / norm_factor;

                  if (std::abs(val) > tol) {
// insert into global array
#pragma omp critical
                    tensor_entries.push_back(tensor_entry_struct_t(
                        tensor_index.i1, tensor_index.i2, tensor_index.i3, val));
                  }
                }
              }
            }  // end for it3
          }    // end for it2
        }      // end for it1
        // end taskgroup

        // Hint: There is a single pragma above, so there is no race condition
        // here despite the parallel region.
        for (auto& elem : entries) {
          entries_multiplicity_offsets.push_back(em_offset_ctr);
          em_offset_ctr += elem.second.size();
          entries_multiplicity.insert(
              entries_multiplicity.end(), elem.second.begin(), elem.second.end());
        }
      }
    }
  }

  // write tensor entries (only unqiue values)
  std::string outfile = vm["out"].as<std::string>();
  std::cout << "writing results to " << outfile << "\n";
  // writes datset `values`
  tensor2hdf(outfile, tensor_entries);
  // append entry duplicates
  // writes datasets `indices` and `offsets`
  entries_multiplicity2hdf(outfile, entries_multiplicity, entries_multiplicity_offsets);

  return 0;
}
