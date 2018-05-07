#pragma once

#include <Eigen/Dense>


namespace boltzmann {

/**
 * @brief Barycentric interpolation formula
 *
 * @param wi
 * @param xi
 */
template <typename NUMERIC = double>
class LagrangePolynomial
{
 public:
  typedef NUMERIC numeric_t;
  typedef Eigen::Matrix<numeric_t, -1, 1> vector_t;

 public:
  /**
   * @brief comput barycentric
   *
   * @param wi   weights from barycentric interpolation formula
   * @param xi   abscissas
   */
  static void compute_weights(vector_t &wi, const vector_t &xi);
  /**
   *
   *
   * @param y    output
   * @param x    evaluation points
   * @param xi   abscissas
   * @param yi   interpolation values
   * @param wi   weights (obtained via compute weights)
   */
  static void evaluate(
      vector_t &y, const vector_t &x, const vector_t &xi, const vector_t &yi, const vector_t &wi);
};

// --------------------------------------------------------------------------------
template <typename NUMERIC>
void
LagrangePolynomial<NUMERIC>::compute_weights(vector_t &wi, const vector_t &xi)
{
  const int N = xi.size();

  assert(wi.size() == xi.size());

  for (int i = 0; i < N; ++i) {
    numeric_t v = 1;
    for (int j = 0; j < N; ++j) {
      if (j != i) v *= (xi[j] - xi[i]);
    }
    wi[i] = 1 / v;
  }
}

// --------------------------------------------------------------------------------
template <typename NUMERIC>
void
LagrangePolynomial<NUMERIC>::evaluate(
    vector_t &y, const vector_t &x, const vector_t &xi, const vector_t &yi, const vector_t &wi)
{
  const numeric_t tol(1e-15);

  vector_t dist(xi.size());

  // number of grid points
  const int N = xi.size();

  // number of evaluation nodes
  const int n = x.size();
  assert(y.size() == x.size());

  // iterate over evaluation points in x(j)
  for (int j = 0; j < n; ++j) {
    // compute distance
    bool done = false;
    for (int k = 0; k < N; ++k) {
      dist(k) = x(j) - xi(k);
      if (std::abs(dist(k)) < tol) {
        y(j) = yi(k);
        done = true;
        break;
      }
    }
    if (done) continue;

    // compute sum
    double denom = 0;
    double numer = 0;

    for (int k = 0; k < N; ++k) {
      double v = (wi[k] / dist[k]);
      denom += v;
      numer += v * yi[k];
    }
    y(j) = numer / denom;
  }
}

}  // end namespace boltzmann
