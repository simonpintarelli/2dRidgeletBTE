#pragma once

#include <cassert>


inline int
pow2p(int j)
{
  assert(j >= 0);
  if (j == 0)
    return 1;
  else
    return 2 << (j - 1);
}
