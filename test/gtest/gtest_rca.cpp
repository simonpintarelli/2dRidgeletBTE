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

TEST(base, rca)
{
  // for (int j = 1; j < 9; ++j) {

  //   RidgeletFrame rf(j, j, 1, 1);

  //   typedef RidgeletCellArray<> rca_t;

  //   rca_t rca(rf);
  //   auto& vcoeffs = rca.coeffs();

  //   auto& lambdas = rf.lambdas();

  //   for (int i = 0; i < vcoeffs.size(); ++i) {
  //     int rows = vcoeffs[i].rows();
  //     int cols = vcoeffs[i].cols();
  //     auto t = tgrid_dim(lambdas[i], rf);
  //     int tx = std::get<0>(t);
  //     int ty = std::get<1>(t);
  //     EXPECT_EQ(rows, tx) << j << lambdas[i];
  //     EXPECT_EQ(cols, ty) << j << lambdas[i];
  //   }
  // }
}
