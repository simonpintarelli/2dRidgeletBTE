#pragma once

#include <exception>
#include <ostream>
#include <string>

#define ASSERT_MSG(cond, msg)                                                                      \
  {                                                                                                \
    if (!(cond)) {                                                                                 \
      throw std::runtime_error(std::string("Error in : ") + __FILE__ + ":" +                       \
                               std::to_string(__LINE__) + ":" + __PRETTY_FUNCTION__ + "\n" + msg); \
    }                                                                                              \
  }

#define ASSERT(cond)                                                                         \
  {                                                                                          \
    if (!(cond)) {                                                                           \
      throw std::runtime_error(std::string("Error in : ") + __FILE__ + ":" +                 \
                               std::to_string(__LINE__) + ":" + __PRETTY_FUNCTION__ + "\n"); \
    }                                                                                        \
  }
