#pragma once

namespace boltzmann {
enum COLLISION_KERNEL
{
  MAXWELLIAN
};

enum METHOD
{
  SUPG,
  LEASTSQUARES,
};

enum TRIG : int
{
  COS = 0,
  SIN = 1
};

enum class KERNEL_TYPE
{
  MAXWELLIAN,
  VHS
};

enum class BC_Type
{
  REGULAR = 0,
  XPERIODIC = 1,
  YPERIODIC = 2,
  XYPERIODIC = 3
};

} // end namespace boltzmann
