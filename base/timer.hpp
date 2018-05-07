#pragma once

#include <stdint.h>
#include <sys/time.h>
#include <time.h>
#include <boost/chrono.hpp>
#include <cassert>
#ifdef HAVE_MPI
#include <mpi.h>
#endif

#include <iomanip>
#include <memory>


extern inline unsigned long long __attribute__((always_inline)) rdtsc()
{
  unsigned int hi, lo;
  __asm__ __volatile__(
      "xorl %%eax, %%eax\n\t"
      "cpuid\n\t"
      "rdtsc"
      : "=a"(lo), "=d"(hi)
      : /* no inputs */
      : "rbx", "rcx");
  return ((unsigned long long)hi << 32ull) | (unsigned long long)lo;
}

/**
 * @brief Measures in CPU cycles
 *
 */
class RDTSCTimer
{
 public:
  RDTSCTimer()
      : started(false)
      , tbegin(0)
  {
  }

  inline void start()
  {
    started = true;
    tbegin = rdtsc();
  }

  inline uint64_t stop()
  {
    uint64_t tend = rdtsc();
    assert(started);
    started = false;
    return tend - tbegin;
  }

  void print(std::ostream &out, uint64_t tlap, const std::string &label)
  {
    out << std::setw(17) << std::left << ("TIMINGS::" + label) << ": " << std::setw(10)
        << std::scientific << std::setprecision(4) << tlap / 1e9 << " [Gcycles]\n";
  }

 private:
  bool started;
  uint64_t tbegin;
};

// ================================================================================
// ================================================================================
// ================================================================================
#ifdef USE_MPI
class BoostTimer
{
 private:
  typedef boost::chrono::time_point<boost::chrono::steady_clock> time_point_t;
  typedef boost::chrono::duration<long long, boost::micro> microseconds;

 public:
  BoostTimer(MPI_Comm comm = MPI_COMM_WORLD)
      : comm_(comm)
  { /* empty */
  }

  inline void start()
  {
    MPI_Barrier(comm_);
    start_ = boost::chrono::high_resolution_clock::now();
  }

  /**
   * @brief returns time in microseconds
   *
   */
  inline long long stop()
  {
    time_point_t now = boost::chrono::high_resolution_clock::now();
    MPI_Barrier(comm_);
    long long global_dt;
    long long dt = boost::chrono::duration_cast<microseconds>(now - start_).count();
    MPI_Allreduce(&dt, &global_dt, 1, MPI_LONG, MPI_MAX, comm_);
    return global_dt;
  }

 private:
  time_point_t start_;
  MPI_Comm comm_;
};

#endif
