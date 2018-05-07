import h5py
from numpy import *

with h5py.File('test.h5', 'r') as fh5:
    A = array(fh5['D'])

nprocs = 4
cs = 100//nprocs

for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        for k in range(A.shape[2]):
            pid = k//cs
            ref = pid*10**4 + 10*k +j
            val = A[i,j,k]
            assert(ref == val)
