#pragma once

#include <fftw3.h>
#include <boost/assert.hpp>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

// own includes
// ----------------------------------------------------------------------
#include "base/hash_specializations.hpp"
#include "base/types.hpp"


/**
 * @brief Handler to manage FFTW plans for real2complex transforms. Before use,
 *        `create_and_get_plan` must be invoked for all sizes / directions
 * required
 *        later. After it was initialized, plans can be queried weith get_plan
 * (thread-safe).
 *
 */
class PlannerR2C
{
 public:
  enum DIR
  {
    FWD,
    INV
  };

 private:
  /// tuple types are: (n0, n1, flags, dir, fft type)
  typedef std::tuple<int, int, unsigned int, int, int> key_t;

 public:
  /**
    * @brief Returns plan (if not already present, it is created, which requires
   * to lock a
    * mutex). This method is thread-safe. WARNING: overhead due to malloc for
   * in, out arrays.
    *
    * @param n
    * @param flags
    * @param dir
    *
    * @return
    */
  fftw_plan create_and_get_plan(int n0, int n1, DIR dir, enum ft_type type = ft_type::R2C);

  /**
   * @brief Returns a plan if available and null otherwise. This method is
   * thread-safe.
   *
   * @param n
   * @param flags
   * @param dir
   *
   * @return
   */
  fftw_plan get_plan(int n[2], DIR dir, enum ft_type type = ft_type::R2C) const;

  /**
   * @brief set fftw flags /
   *
   * @param flags
   */
  void set_flags(unsigned int flags)
  {
    if (flags != flags_) plans_.clear();
    flags_ = flags;
  }

  virtual ~PlannerR2C()
  {
    for (auto &v : plans_) {
      fftw_destroy_plan(v.second);
    }
  }

  void print(std::ostream &out = std::cout) const;

 protected:
  unsigned int flags_ = FFTW_ESTIMATE;
  std::unordered_map<key_t, fftw_plan> plans_;
  std::mutex mutex_;
};

// -------------------------------------------------------------------------------------
inline fftw_plan
PlannerR2C::create_and_get_plan(int n0, int n1, DIR dir, enum ft_type type)
{
  // the only thread safe routine in fftw is fftw_execute, thus we need a lock
  // here
  std::lock_guard<std::mutex> lock(mutex_);

  // const int n0 = n[0];
  // const int n1 = n[1];
  int n[2] = {n0, n1};
  int embed[2] = {n0, n1};
  key_t key = std::make_tuple(n0, n1, flags_, int(dir), int(type));
  auto it = plans_.find(key);

  BOOST_ASSERT_MSG((dir == FWD) || (dir == INV), "invalid direction flag");

  if (it != plans_.end()) {
    return it->second;
  } else {
    if (type == ft_type::R2C) {
      if (dir == FWD) {
        fftw_complex *out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * n[0] * n[1]);
        double *in = (double *)fftw_malloc(sizeof(double) * n[0] * n[1]);
        fftw_plan fwd_plan = fftw_plan_many_dft_r2c(2 /* rank */,
                                                    n /* dims */,
                                                    1 /* num dfts */,
                                                    in,
                                                    embed,
                                                    1 /* stride */,
                                                    embed[0] * embed[1],
                                                    out,
                                                    embed,
                                                    1 /*stride */,
                                                    embed[0] * embed[1],
                                                    flags_);
        fftw_free(out);
        fftw_free(in);
        BOOST_ASSERT_MSG(fwd_plan != NULL, "FFTW failed to create plan");
        plans_[key] = fwd_plan;
        return fwd_plan;

      } else if (dir == INV) {
        fftw_complex *out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * n[0] * n[1]);
        double *in = (double *)fftw_malloc(sizeof(double) * n[0] * n[1]);
        fftw_plan inv_plan = fftw_plan_many_dft_c2r(2 /* rank */,
                                                    n /* dims */,
                                                    1 /* num dfts */,
                                                    out,
                                                    embed,
                                                    1 /* stride */,
                                                    embed[0] * embed[1],
                                                    in,
                                                    embed,
                                                    1 /*stride */,
                                                    embed[0] * embed[1],
                                                    flags_);
        fftw_free(out);
        fftw_free(in);
        BOOST_ASSERT_MSG(inv_plan != NULL, "FFTW failed to create plan");
        plans_[key] = inv_plan;
        return inv_plan;
      }
    } else if (type == ft_type::C2C) {
      if (dir == FWD) {
        fftw_complex *out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * n[0] * n[1]);
        fftw_complex *in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * n[0] * n[1]);
        fftw_plan fwd_plan = fftw_plan_dft_2d(n0, n1, in, out, FFTW_FORWARD, flags_);
        fftw_free(out);
        fftw_free(in);
        BOOST_ASSERT_MSG(fwd_plan != NULL, "FFTW failed to create plan");
        plans_[key] = fwd_plan;
        return fwd_plan;
      } else if (dir == INV) {
        fftw_complex *out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * n[0] * n[1]);
        fftw_complex *in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * n[0] * n[1]);
        fftw_plan inv_plan = fftw_plan_dft_2d(n0, n1, in, out, FFTW_BACKWARD, flags_);
        fftw_free(out);
        fftw_free(in);
        BOOST_ASSERT_MSG(inv_plan != NULL, "FFTW failed to create plan");
        plans_[key] = inv_plan;
        return inv_plan;
      }
    } else {
      throw std::runtime_error("unknown fft type");
    }
  }
}

// --------------------------------------------------------------------------------
inline fftw_plan
PlannerR2C::get_plan(int n[2], DIR dir, enum ft_type type) const
{
  auto it = plans_.find(std::make_tuple(n[0], n[1], flags_, int(dir), int(type)));
  if (it != plans_.end())
    return it->second;
  else
    return NULL;
}

// --------------------------------------------------------------------------------
inline void
PlannerR2C::print(std::ostream &out) const
{
  for (auto it = plans_.begin(); it != plans_.end(); ++it) {
    auto key = it->first;
    int nx = std::get<0>(key);
    int ny = std::get<1>(key);
    unsigned int flags = std::get<2>(key);
    int dir = std::get<3>(key);
    int type = std::get<4>(key);

    std::map<int, std::string> flag2names;
    std::map<int, std::string> type2names;

    flag2names[FFTW_ESTIMATE] = "FFTW_ESTIMATE";
    flag2names[FFTW_MEASURE] = "FFTW_MEASURE";
    flag2names[FFTW_ESTIMATE_PATIENT] = "FFTW_ESTIMATE_PATIENT";
    flag2names[FFTW_EXHAUSTIVE] = "FFTW_EXHAUSTIVE";

    type2names[int(ft_type::C2C)] = "C2C";
    type2names[int(ft_type::R2C)] = "R2C";

    if (dir == FWD)
      out << "FWD: " << type2names[type] << " " << std::setw(10) << nx << " " << std::setw(10) << ny
          << ": " << flag2names.at(flags) << std::endl;
    else if (dir == INV)
      out << "INV: " << type2names[type] << " " << std::setw(10) << nx << " " << std::setw(10) << ny
          << ": " << flag2names.at(flags) << std::endl;
    else
      out << "FFTW PLANNER: DIR NOT FOUND!\n";
  }
}

// =====================================================================================
// =====================================================================================
// =====================================================================================
/**
 * @brief Creates plans upon request. Every call to get_plan invovles obtaining
 * a lock.
 *        This class and all it's methods are thread-safe.
 *
 */
class PlannerR2COD : public PlannerR2C
{
 public:
  /**
   * @brief WARNING: overhead due to malloc for in, out arrays.
   *
   * @param n   [nx, ny]
   * @param dir cf. PlannerR2C::DIR
   *
   * @return fftw_plan
   */
  fftw_plan get_plan(int n[2], DIR dir, enum ft_type type = ft_type::R2C)
  {
    return PlannerR2C::create_and_get_plan(n[0], n[1], dir, type);
  }

  fftw_plan get_plan(int n[2], DIR dir, enum ft_type type = ft_type::R2C) const = delete;
};
