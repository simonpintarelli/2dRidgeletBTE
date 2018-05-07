#!/bin/env python

from numpy import *
from pyboltz.basis import KSBasis
import re, os
from libSpectralTools import *
import h5py
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-K', type=int, help='polynomial deg.', default=20)

    args =  parser.parse_args()
    K = args.K

    p2n = Polar2Nodal(K, 0.5)
    shift = ShiftPolar(K, 0.5)
    basis = KSBasis(K=K, w=0.5)

    C =  zeros(len(basis))
    C[0] = 1/2/pi
    Cleft = zeros_like(C)
    Ctop = zeros_like(C)
    shift.shift(Cleft, C, -3, 0)
    shift.shift(Ctop, C, 0, -3)

    cn = zeros((K, K))
    p2n.to_nodal(cn, C)
    qleft = zeros((K, K))
    p2n.to_nodal(qleft, Cleft)
    qtop = zeros((K, K))
    p2n.to_nodal(qtop, Ctop)


    fname = 'init_inflow.h5'
    fh5 = h5py.File(fname, 'w')
    fh5.create_dataset('qleft', data=qleft, shape=qleft.shape)
    fh5.create_dataset('qtop', data=qtop, shape=qtop.shape)
    fh5.create_dataset('rho', data=cn, shape=cn.shape)
    fh5.close()

    print('Written file `%s`.' % fname)
