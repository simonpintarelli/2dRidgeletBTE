#pragma once

#include <boost/assert.hpp>
#include <boost/lexical_cast.hpp>

#include "base/buffered_output_array.hpp"
#include "base/coeff_array.hpp"
#include "export/hdf_writer.hpp"
#include "spectral/macroscopic_quantities.hpp"
#include "spectral/polar_to_nodal.hpp"


class OutputHandler
{
 private:
  template <int dim>
  using buffer_t = BufferType<dim>;

 public:
  template <typename POLAR_BASIS>
  OutputHandler(const POLAR_BASIS &basis,
                const Map &xmap,
                const boltzmann::Polar2Nodal<POLAR_BASIS> &p2n,
                int freq_mom = -1,
                int dump = -1,
                int bufsize = 1);

  void compute(const CoeffArray<double> &full_vector, unsigned int timestep, double time);

 private:
  /// DoF distribution
  Map xmap_;
  /// Polar2Nodal
  const boltzmann::Polar2Nodal<> p2n_;
  /// moments output frequency
  int freq_mom_;
  /// solution output frequency (full solution vector)
  int dump_;
  /// buffer size
  int bufsize_;
  /// buffer counter
  int ctr_ = 0;
  /// density (scalar)
  buffer_t<1> m_;
  /// energy (scalar)
  buffer_t<1> e_;
  /// velocity (vector)
  buffer_t<3> v_;
  /// heat flow (vector)
  buffer_t<3> q_;
  /// energy flow (vector)
  buffer_t<3> r_;
  /// momentum flow  (tensor)
  buffer_t<9> M_;
  /// pressure (tensor)
  buffer_t<9> P_;
  /// object to compute macroscopic quantities
  boltzmann::MQEval mq_coeffs_;

  std::vector<double> timesteps_;
};

template <typename POLAR_BASIS>
OutputHandler::OutputHandler(const POLAR_BASIS &basis,
                             const Map &xmap,
                             const boltzmann::Polar2Nodal<POLAR_BASIS> &p2n,
                             int freq_mom,
                             int dump,
                             int bufsize)
    : xmap_(xmap)
    , p2n_(p2n)
    , freq_mom_(freq_mom)
    , dump_(dump)
    , bufsize_(bufsize)
    , m_(xmap, bufsize)
    , e_(xmap, bufsize)
    , v_(xmap, bufsize)
    , q_(xmap, bufsize)
    , r_(xmap, bufsize)
    , M_(xmap, bufsize)
    , P_(xmap, bufsize)
{
  mq_coeffs_.init(basis);
}

void
OutputHandler::compute(const CoeffArray<double> &full_vector, unsigned int timestep, double time)
{
  typedef typename CoeffArray<double>::map_t map_t;
  typedef map_t::index_t index_t;

  BOOST_ASSERT(full_vector.get_map() == xmap_);

  if (timestep % freq_mom_ == 0) {
    timesteps_.push_back(time);
// check if buffer is full? => write
#pragma omp parallel
    {
      // boltzmann::Polar2Nodal_Evaluator<> p2n_adapter(p2n_);
      // obtain thread-safe evaluator
      auto evaluator = mq_coeffs_.evaluator();
      Eigen::ArrayXd cp(p2n_.N());
      int K = p2n_.K();

#pragma omp for schedule(static)
      for (index_t lid = 0; lid < full_vector.get_map().lsize(); ++lid) {
        // TODO: call p2n, reshape full_vector.get(lid) first
        typedef Eigen::MatrixXd matrix_t;
        Eigen::Map<const matrix_t> cn(full_vector.get(lid).data(), K, K);
        p2n_.to_polar(cp, cn);
        double norm = cp.abs2().sum();
        evaluator(cp);
        // access results and copy to buffer
        m_.fill(evaluator.m, ctr_, lid);
        e_.fill(evaluator.e, ctr_, lid);
        v_.fill(evaluator.v, ctr_, lid);
        q_.fill(evaluator.q, ctr_, lid);
        r_.fill(evaluator.r, ctr_, lid);
        M_.fill(evaluator.M, ctr_, lid);
        P_.fill(evaluator.P, ctr_, lid);
      }
    }
    ctr_++;  // increment buffer counter
    if (ctr_ == bufsize_) {
      // write HDF
      char fname[256];
      std::sprintf(fname, "solution_vector%06d.h5", timestep);
      PHDFWriter exporter(fname, full_vector.get_map().comm());
      exporter.write(m_, "m");
      exporter.write(e_, "e");
      exporter.write(v_, "v");
      exporter.write(q_, "q");
      exporter.write(r_, "r");
      exporter.write(M_, "M");
      exporter.write(P_, "P");
      // todo write meta data ...
      // reset counter
      ctr_ = 0;
    }
  }

  if (timestep % dump_ == 0) {
    char fname[256];
    std::sprintf(fname, "fsolution_vector%06d.h5", timestep);
    PHDFWriter exporter(fname, full_vector.get_map().comm());
    exporter.write(full_vector, "C");
  }
}
