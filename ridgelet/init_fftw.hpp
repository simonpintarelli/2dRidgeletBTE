#pragma once

#include <type_traits>
#include <vector>
#include "base/pow2p.hpp"
#include "base/types.hpp"
#include "fft/planner.hpp"
#include "lambda.hpp"


template <template <class> class FFT, typename PLAN_HANDLER, typename FRAME>
void
init_fftw(FFT<PLAN_HANDLER> &fft, unsigned int flags, const FRAME &frame)
{
  typedef PLAN_HANDLER planner_t;

  if (std::is_same<planner_t, PlannerR2COD>::value) {
    // nothing todo, since plans are created on demand
    return;
  }

  const int rho_x = frame.rho_x();
  const int rho_y = frame.rho_y();
  const int Jx = frame.Jx();
  const int Jy = frame.Jy();
  unsigned int Nx = frame.Nx();
  unsigned int Ny = frame.Ny();

  auto &lambdas = frame.lambdas();
  auto &planner = fft.get_plan();

  planner.set_flags(flags);

  for (unsigned int i = 0; i < frame.size(); ++i) {
    auto lam = lambdas[i];
    const int j = lam.j;
    if (lam.t == rt_type::X) {
      planner.create_and_get_plan(8 * rho_y, pow2p(j + 2) * rho_x, planner_t::INV, ft_type::R2C);
      planner.create_and_get_plan(8 * rho_y, pow2p(j + 2) * rho_x, planner_t::INV, ft_type::C2C);
      planner.create_and_get_plan(
          4 * (std::abs(lam.k) + 1) * rho_y, pow2p(j + 2) * rho_x, planner_t::FWD, ft_type::R2C);
      planner.create_and_get_plan(
          4 * (std::abs(lam.k) + 1) * rho_y, pow2p(j + 2) * rho_x, planner_t::FWD, ft_type::C2C);
    } else if (lam.t == rt_type::Y) {
      planner.create_and_get_plan(pow2p(j + 2) * rho_y, 8 * rho_x, planner_t::INV, ft_type::R2C);
      planner.create_and_get_plan(pow2p(j + 2) * rho_y, 8 * rho_x, planner_t::INV, ft_type::C2C);
      planner.create_and_get_plan(
          pow2p(j + 2) * rho_y, 4 * (std::abs(lam.k) + 1) * rho_x, planner_t::FWD, ft_type::R2C);
      planner.create_and_get_plan(
          pow2p(j + 2) * rho_y, 4 * (std::abs(lam.k) + 1) * rho_x, planner_t::FWD, ft_type::C2C);
    } else if (lam.t == rt_type::D) {
      planner.create_and_get_plan(8 * rho_x, pow2p(j + 2) * rho_y, planner_t::INV, ft_type::R2C);
      planner.create_and_get_plan(8 * rho_x, pow2p(j + 2) * rho_y, planner_t::INV, ft_type::C2C);
      planner.create_and_get_plan(
          pow2p(j + 2) * rho_x, pow2p(j + 2) * rho_y, planner_t::FWD, ft_type::R2C);
      planner.create_and_get_plan(
          pow2p(j + 2) * rho_x, pow2p(j + 2) * rho_y, planner_t::FWD, ft_type::C2C);
    } else if (lam.t == rt_type::S) {
      planner.create_and_get_plan(4 * rho_y, 4 * rho_x, planner_t::INV, ft_type::R2C);
      planner.create_and_get_plan(4 * rho_y, 4 * rho_x, planner_t::INV, ft_type::C2C);
      planner.create_and_get_plan(4 * rho_y, 4 * rho_x, planner_t::FWD, ft_type::R2C);
      planner.create_and_get_plan(4 * rho_y, 4 * rho_x, planner_t::FWD, ft_type::C2C);
    }
  }

  int J = std::max(Jx, Jy);

  planner.create_and_get_plan(
      pow2p(J + 2) * rho_y, pow2p(J + 2) * rho_y, planner_t::INV, ft_type::R2C);
  planner.create_and_get_plan(
      pow2p(J + 2) * rho_y, pow2p(J + 2) * rho_y, planner_t::INV, ft_type::C2C);
  planner.create_and_get_plan(
      pow2p(J + 2) * rho_y, pow2p(J + 2) * rho_y, planner_t::FWD, ft_type::R2C);
  planner.create_and_get_plan(
      pow2p(J + 2) * rho_y, pow2p(J + 2) * rho_y, planner_t::FWD, ft_type::C2C);

  planner.create_and_get_plan(Ny / 2, Nx / 2, planner_t::INV, ft_type::R2C);
  planner.create_and_get_plan(Ny / 2, Nx / 2, planner_t::INV, ft_type::C2C);
  planner.create_and_get_plan(Ny / 2, Nx / 2, planner_t::FWD, ft_type::R2C);
  planner.create_and_get_plan(Ny / 2, Nx / 2, planner_t::FWD, ft_type::C2C);
}
