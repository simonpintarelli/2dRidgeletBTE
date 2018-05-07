#include "base/numbers.hpp"

#include "qmidpoint.hpp"

namespace boltzmann {

QMidpoint::QMidpoint(int npts)
    : base_type(npts)
{
  double h = 2 * numbers::PI / (1.0 * npts);

  for (int i = 0; i < npts; ++i) {
    this->pts_[i] = i * h;
    this->wts_[i] = h;
  }
}

}  // end namespace boltzmann
