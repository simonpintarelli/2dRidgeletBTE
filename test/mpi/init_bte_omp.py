#!/usr/bin/env python
import matplotlib as mpl
mpl.use('Agg')
from numpy import *
from numpy.linalg import norm
from pyboltz.basis import get_basis, KSBasis
from pyboltz.quadrature import TensorQuadFactory
from pyboltz.observables import Mass, Momentum, Energy
from pyboltz.initialize import project_to_polar_basis, Maxwellian
from spectral_tools import Polar2Hermite, Polar2Nodal, ShiftPolar
from matplotlib import ticker
from bquad import MaxwellQuad
import h5py
import os
import sys
import glob
import argparse

from pyFFRT import *
from matplotlib.pyplot import *
# ===============================================
#                    vel. dofs.
#            +----------------------+
#            |                      |
#            |                      |
#            |                      | spatial
#            |                      |  dofs
#            |                      |
#            |                      |
#            +----------------------+
#
#  the velocity DoFs are flattened in fortran order
#

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-j', type=int)
    parser.add_argument('-k', type=int, default=20)
    args = parser.parse_args()
    J = args.j + 2
    k = args.k
    Nx = 2**(J-1)
    print('Nx: ', Nx)
    spectral_basis = KSBasis(K=20, w=0.5)
    spectral_basis.write_desc('spectral_basis.desc')
    print('using basis of size ', len(spectral_basis))
    N = len(spectral_basis)
    K = spectral_basis.get_K()
    p2h = Polar2Hermite(K, 0.5)
    p2n = Polar2Nodal(K, 0.5)

    def to_nodal(cp):
        """
        """
        C = zeros((K, K))
        p2n.to_nodal(C, cp)
        return C

    debug = False  # check that dof ordering is correct
    # num levels (Ridgelets)
    xi = linspace(0, 1, Nx, endpoint=False)
    yi = linspace(0, 1, Nx, endpoint=False)
    nx = len(xi)
    ny = len(yi)

    X, Y = meshgrid(xi, yi)
    Z = stack((X, Y), axis=2)

    C = zeros(N)
    C[0] = 1 / np.pi / 2
    quad = MaxwellQuad(0.5, 110)
    mass = Mass(spectral_basis, quad)
    momentum = Momentum(spectral_basis, quad)
    rho = mass.compute(C)
    print('mass: %f' % rho)

    CP1 = zeros_like(C)
    CP2 = zeros_like(C)

    v0 = 4
    shift_polar = ShiftPolar()
    shift_polar.shift(CP1, C, -v0, -v0)
    shift_polar.shift(CP2, C, v0, v0)

    u1p = momentum.compute(CP1)
    u2p = momentum.compute(CP2)

    u1 = -v0 - 1j*v0
    u2 = v0 + 1j*v0

    print('u1p: ', u1p)
    print('u2p: ', u2p)

    T = 1
    # M1 = Maxwellian(m=1, e=2*T, u=u2)
    # M2 = Maxwellian(m=1, e=2*T, u=u1)
    # CP1 = project_to_polar_basis(M1, spectral_basis, quad_handler, 100, 100)
    # CP2 = project_to_polar_basis(M2, spectral_basis, quad_handler, 100, 100)

    CN1 = to_nodal(CP1)
    CN2 = to_nodal(CP2)
    CN = to_nodal(C)

    fx = lambda x, c: 1 * exp(-200 * norm(x - c.reshape(1, 1, 2), axis=2)**2)
    F1 = fx(Z, array([0.4, 0.4]))
    F2 = fx(Z, array([0.6, 0.6]))

    print('F1.shape: ', F1.shape)

    C = (outer(0.3*ones(prod(F1.shape)), CN.flatten(order='F'))
         + outer(F1.flatten(), CN1.flatten(order='F'))
         + outer(F2.flatten(), CN2.flatten(order='F')))

    assert (C.shape == (prod(F1.shape), prod(CN1.shape)))

    with h5py.File('output_handler_init.h5', 'w') as fh5:
        fh5.create_dataset('coeffs', data=C, shape=C.shape)
