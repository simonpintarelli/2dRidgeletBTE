#include <gtest/gtest.h>
#include <boost/lexical_cast.hpp>

#include "base/eigen2hdf.hpp"
#include "fft/fft2.hpp"
#include "fft/fft2_r2c.hpp"
#include "ridgelet/construction/translation_grid.hpp"
#include "ridgelet/ridgelet_cell_array.hpp"
#include "ridgelet/ridgelet_frame.hpp"
#include "ridgelet/rt.hpp"

using namespace std;

/**
 *  @brief test rt -> irt
 *
 *  performs tests for PlannerR2COD
 *
 *  @param param
 *  @return return type
 */
TEST(base, rt)
{
  typedef RT<double, RidgeletFrame, FFTr2c<PlannerR2COD> > rt_t;
  typedef typename rt_t::numeric_t numeric_t;
  typedef typename rt_t::complex_array_t complex_array_t;
  typedef typename rt_t::fft_t fft_t;
  typedef typename rt_t::rt_coeff_t rt_coeff_t;
  fft_t fft;
  double tol = 1e-15;

  int J = 7;

  unsigned int rho_x = 1;
  unsigned int rho_y = 1;

  for (int j = 0; j < J; ++j) {
    RidgeletFrame rf(J, J, rho_x, rho_y);
    unsigned int Nx = rf.Nx();
    unsigned int Ny = rf.Ny();
    rt_t rt(rf);

    typedef Eigen::Array<numeric_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> array_t;
    array_t R(Nx, Ny);
    R.setRandom();

    std::vector<rt_coeff_t> rt_coeffs(rf.size());

    complex_array_t Fh(Nx, Ny);
    fft.ft(Fh, R);
    rt.rt(rt_coeffs, Fh);

    // check that tgrid_dim is correct
    auto& lambdas = rf.lambdas();
    for (unsigned int i = 0; i < lambdas.size(); ++i) {
      auto tt = tgrid_dim(lambdas[i], rf);
      int rows = rt_coeffs[i].rows();
      int cols = rt_coeffs[i].cols();

      EXPECT_EQ(rows * cols, std::get<0>(tt) * std::get<1>(tt)) << lambdas[i];
      // EXPECT_EQ(rows, std::get<0>(tt)) << lambdas[i];
      // EXPECT_EQ(cols, std::get<1>(tt)) << lambdas[i];
    }

    complex_array_t Fh2(Nx, Ny);
    rt.irt(Fh2, rt_coeffs);

    // check that is invertible
    array_t DIFF = (ftcut(Fh2, Nx / 2, Ny / 2) - ftcut(Fh, Nx / 2, Ny / 2)).abs();
    double dmax = DIFF.maxCoeff();
    EXPECT_TRUE(dmax < tol) << dmax;
  }
}

/**
 *  @brief test rt -> irt
 *
 *  performs tests for PlannerR2COD
 *
 *  @param param
 *  @return return type
 */
TEST(base, rt_planned)
{
  typedef RT<double, RidgeletFrame, FFTr2c<PlannerR2C> > rt_t;
  typedef typename rt_t::numeric_t numeric_t;
  typedef typename rt_t::complex_array_t complex_array_t;
  typedef typename rt_t::fft_t fft_t;
  typedef typename rt_t::rt_coeff_t rt_coeff_t;
  fft_t fft;
  double tol = 1e-15;

  int J = 7;

  unsigned int rho_x = 1;
  unsigned int rho_y = 1;

  for (int j = 0; j < J; ++j) {
    RidgeletFrame rf(J, J, rho_x, rho_y);
    unsigned int Nx = rf.Nx();
    unsigned int Ny = rf.Ny();
    rt_t rt(rf);
    init_fftw(fft, FFTW_MEASURE, rf);

    typedef Eigen::Array<numeric_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> array_t;
    array_t R(Nx, Ny);
    R.setRandom();

    std::vector<rt_coeff_t> rt_coeffs(rf.size());

    complex_array_t Fh(Nx, Ny);
    fft.ft(Fh, R);
    rt.rt(rt_coeffs, Fh);

    // check that tgrid_dim is correct
    auto& lambdas = rf.lambdas();
    for (unsigned int i = 0; i < lambdas.size(); ++i) {
      auto tt = tgrid_dim(lambdas[i], rf);
      int rows = rt_coeffs[i].rows();
      int cols = rt_coeffs[i].cols();

      EXPECT_EQ(rows * cols, std::get<0>(tt) * std::get<1>(tt)) << lambdas[i];
      // EXPECT_EQ(rows, std::get<0>(tt)) << lambdas[i];
      // EXPECT_EQ(cols, std::get<1>(tt)) << lambdas[i];
    }

    complex_array_t Fh2(Nx, Ny);
    rt.irt(Fh2, rt_coeffs);

    // check that is invertible
    array_t DIFF = (ftcut(Fh2, Nx / 2, Ny / 2) - ftcut(Fh, Nx / 2, Ny / 2)).abs();
    double dmax = DIFF.maxCoeff();
    EXPECT_TRUE(dmax < tol) << dmax;
  }
}




/**
 *  @brief check that rt transform is invertible if the Fourier coefficients are padded by zeros.
 *  (the zero-padding is done to have invertibility on the entire grid)
 *  Detailed description
 *
 *  @param param
 *  @return return type
 */
TEST(base, rt_padded)
{
  typedef RT<double, RidgeletFrame, FFTr2c<PlannerR2COD> > rt_t;
  typedef typename rt_t::numeric_t numeric_t;
  typedef typename rt_t::complex_array_t complex_array_t;
  typedef typename rt_t::fft_t fft_t;
  typedef typename rt_t::rt_coeff_t rt_coeff_t;
  fft_t fft;
  double tol = 1e-15;

  int J = 6;

  unsigned int rho_x = 1;
  unsigned int rho_y = 1;

  for (int j = 0; j < J; ++j) {
    RidgeletFrame rf(J, J, rho_x, rho_y);
    unsigned int Nx = rf.Nx();
    unsigned int Ny = rf.Ny();
    unsigned int nx = Nx / 2;
    unsigned int ny = Ny / 2;
    rt_t rt(rf);

    typedef Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> array_t;
    array_t R(nx, ny);
    R.setRandom();

    std::vector<rt_coeff_t> rt_coeffs(rf.size());

    complex_array_t Fh0(nx, ny);
    fft.ft(Fh0, R);
    // set lowest frequency to zero (e.g. make sure that the Fourier series is
    // real-valued also on the fine grid)
    hf_zero(Fh0);

    complex_array_t Fh(Nx, Ny);
    Fh.setZero();
    ftcut(Fh, nx, ny) = Fh0;

    rt.rt(rt_coeffs, Fh);

    // check that tgrid_dim is correct
    auto& lambdas = rf.lambdas();
    for (unsigned int i = 0; i < lambdas.size(); ++i) {
      auto tt = tgrid_dim(lambdas[i], rf);
      int rows = rt_coeffs[i].rows();
      int cols = rt_coeffs[i].cols();

      EXPECT_EQ(rows * cols, std::get<0>(tt) * std::get<1>(tt)) << lambdas[i];

      // EXPECT_EQ(rows, std::get<0>(tt)) << lambdas[i];
      // EXPECT_EQ(cols, std::get<1>(tt)) << lambdas[i];
    }

    complex_array_t Fhp(Nx, Ny);
    rt.irt(Fhp, rt_coeffs);

    complex_array_t Fhp0(nx, ny);
    Fhp0 = ftcut(Fhp, nx, ny);

    std::string fname = "base_rt_padded" + boost::lexical_cast<std::string>(j) + ".h5";
    hid_t h5f = H5Fcreate(fname.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    eigen2hdf::save(h5f, "Fh0", Fh0);
    eigen2hdf::save(h5f, "Fh", Fh);
    eigen2hdf::save(h5f, "Fhp", Fhp);
    H5Fclose(h5f);

    // check that is invertible
    array_t DIFF = (Fhp0 - Fh0).abs();
    double dmax = DIFF.maxCoeff();
    EXPECT_TRUE(dmax < tol) << dmax;
  }
}

// todo check irt, when ft coeffs are padded
