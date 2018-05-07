#pragma once

#include <complex>


namespace boltzmann {

// forward declaration
class XiR;
class XiRC;
class LaguerreRR;
class LaguerreKS;
class HermiteH;

template <typename T>
struct SpectralCoordTraits
{
};

template <>
struct SpectralCoordTraits<XiR>
{
  typedef double return_type;
  typedef double coord_type;
};

template <>
struct SpectralCoordTraits<XiRC>
{
  typedef std::complex<double> return_type;
  typedef double coord_type;
};

template <>
struct SpectralCoordTraits<LaguerreRR>
{
  typedef double return_type;
  typedef double coord_type;
};

template <>
struct SpectralCoordTraits<LaguerreKS>
{
  typedef double return_type;
  typedef double coord_type;
};

template <>
struct SpectralCoordTraits<HermiteH>
{
  typedef double return_type;
  typedef double coord_type;
};

}  // end namespace boltzmann
