#pragma once

#include "brt_config.h"


#define SOURCE_INFO()                                               \
  std::cout << "SOURCE::INFO::EXEC " << argv[0] << "\n";            \
  std::cout << "SOURCE::INFO::GIT_BRANCHNAME " << GIT_BNAME << "\n" \
            << "SOURCE::INFO::GIT_SHA1       " << GIT_SHA1 << "\n"  \
            << "SOURCE::INFO::GIT_DESCRIBE   " << GIT_DT << "\n";
