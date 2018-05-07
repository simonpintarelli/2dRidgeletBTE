#pragma once

#include "base/csingleton.hpp"

#include "base/hash_specializations.hpp"

#include <fftw3.h>
#include <unordered_map>


/**
 * @brief warning: change this in the future, not all parameters are represented
 * by key
 *
 * @param n0
 * @param n1
 * @param type
 *
 * @return
 */
class RealFFT_Planner2D : public CSingleton<RealFFT_Planner2D>
{
 public:
  enum class rfft_type
  {
    C2R,
    R2C
  };

 private:
  typedef int rfft_t;

 public:
  typedef std::tuple<int, int, rfft_t> key_t;

  /**
   *
   *
   * @param n0
   * @param n1
   *
   * @return
   */
  fftw_plan get_plan(int n0, int n1, rfft_type type)
  {
    auto key = std::make_tuple(n0, n1, rfft_t(type));
    auto it = plans_.find(key);

    if (it == plans_.end()) {
      return NULL;
    } else {
      return it->second;
    }
  }

  void add_plan(int n0, int n1, rfft_type type, fftw_plan plan)
  {
    plans_[std::make_tuple(n0, n1, rfft_t(type))] = plan;
  }

 private:
  std::unordered_map<key_t, fftw_plan> plans_;
};
