#pragma once

/**
 * @brief polyval (Horner scheme)
 *
 * @param coeffs coefficients in ascending order, i.e. the last entry is the
 * highest degree.
 * @param x
 * @param N length of coeffs
 */
template <typename NUMERIC1, typename NUMERIC2>
inline NUMERIC2
polyval(NUMERIC1 *coeffs, NUMERIC2 x, int N)
{
  typedef NUMERIC2 numeric_t;
  numeric_t b = coeffs[N - 1];
  for (int i = 1; i < N - 1; ++i) {
    b = coeffs[N - 1 - i] + b * x;
  }
  return x * b + coeffs[0];
}
