#pragma once


#ifdef USE_MPI
#include <mpi.h>
#endif
#include <boost/assert.hpp>
#include <boost/lexical_cast.hpp>
#include <fstream>
#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <string>

#include "base/exceptions.hpp"
#include "csingleton.hpp"

class Logger : public CSingleton<Logger>
{
 public:
  Logger()
  {
#ifdef USE_MPI
    int pid = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    ASSERT(pid >= 0);
    stream_ptr_ =
        std::make_shared<std::ofstream>("out_" + boost::lexical_cast<std::string>(pid) + ".log");
#else
    stream_ptr_ = std::make_shared<std::ofstream>("out_.log");
#endif
  }

  template <typename T>
  Logger &operator<<(const T &output)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    sbuf_.str("");

    for (auto &v : prefixes_) {
      sbuf_ << v << "::";
    }
    sbuf_ << "\t" << output << std::endl;
    if (stream_ptr_) {
      auto &out = *(stream_ptr_.get());
      out << sbuf_.str();
    }

    // output to console & file
    if (!detach_stdout_) std::cout << sbuf_.str();

    return *this;
  }

  void push_prefix(const std::string &tag)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    prefixes_.push_back(tag);
  }

  void pop_prefix()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    prefixes_.pop_back();
  }

  void clear_prefix()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    prefixes_.clear();
  }

  void flush()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    // make sure to write intermediate changes
    auto &out = *(stream_ptr_.get());
    out.flush();
  }

  void detach_stdout() { detach_stdout_ = true; }

  void attach_stdout() { detach_stdout_ = false; }

 private:
  std::list<std::string> prefixes_;
  std::shared_ptr<std::ostream> stream_ptr_;
  std::mutex mutex_;
  std::stringstream sbuf_;
  bool detach_stdout_ = false;
};
