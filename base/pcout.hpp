#pragma once

#ifdef USE_MPI
#include <mpi.h>
#endif
#include <mutex>


class ConditionalOstream
{
 public:
  template <typename T>
  ConditionalOstream &operator<<(const T &output)
  {
    std::lock_guard<std::mutex> lock(mutex_);
#ifdef USE_MPI
    int pid;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    if (pid == 0) {
      std::cout << output;
    }
#else
    std::cout << output;
#endif
    return *this;
  }

 protected:
  std::mutex mutex_;
};

static ConditionalOstream pcout;
