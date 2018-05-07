#include <gtest/gtest.h>
#include <boost/lexical_cast.hpp>

#include "base/eigen2hdf.hpp"
#include "fft/fft2.hpp"
#include "fft/fft2_r2c.hpp"
#include "ridgelet/rc_linearize.hpp"
#include "ridgelet/ridgelet_cell_array.hpp"
#include "ridgelet/ridgelet_frame.hpp"
#include "ridgelet/rt.hpp"

using namespace std;

TEST(base, rca)
{
  for (int j = 0; j < 9; ++j) {
    RidgeletFrame rf(j, j, 1, 1);

    typedef RidgeletCellArray<> rca_t;

    rca_t rca(rf);
    auto& vcoeffs = rca.coeffs();
    RCLinearize rcl(rf);
  }
}
