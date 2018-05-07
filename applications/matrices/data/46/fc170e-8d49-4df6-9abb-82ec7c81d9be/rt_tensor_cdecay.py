import h5py
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pylab as plt
import itertools

def compute_decay(indices, offsets, values):
    """
    Keyword Arguments:
    indices --
    offsets --
    values  --

    Returns:
    (csum, values_sorted)
    """

    mcount = {}
    for b, e in zip(offsets[:-1], offsets[1:]):
        k = tuple(indices[b])
        mcount[k] = e-b

    values['v'] = np.abs(values['v'])
    values_sorted = np.sort(values, order='v')[::-1]

    counts = np.array([mcount[tuple(b)] for b in values_sorted[['i1', 'i2', 'i3']]], dtype=int)
    csum = np.cumsum(counts)

    return csum, values_sorted


## -------------------- main --------------------

# load data
with h5py.File('rt_tensor.h5','r') as fh5:
    indices = np.array(fh5['indices'])
    offsets = np.array(fh5['offsets'])
    values  = np.array(fh5['values'])

offsets = np.hstack((offsets, np.array(len(indices), dtype=np.uint64)))

# filter data
indices_filtered = []
for b, e in zip(offsets[:-1], offsets[1:]):
    lindices = list(filter(lambda x: x[0] >= x[1] and x[1] >= x[2], indices[b:e]))
    indices_filtered.append(lindices)
offsets_filtered = np.hstack((np.array([0]), np.cumsum(list(map(len, indices_filtered)))))
# flatten indices_filtered after computing offsets, flatten indices_filtered
indices_filtered = np.array(list(itertools.chain.from_iterable(indices_filtered)))

# filter also values
values_filtered = np.array(list(filter(lambda x: x[0] >= x[1] and x[1] >= x[2],
                                       values)))

# overwrite rt_tensor.h5??
csum, values_sorted = compute_decay(indices_filtered, offsets_filtered, values_filtered)

# plot results
plt.figure()
plt.semilogy(csum, values_sorted['v'])
plt.savefig('cdecay.png')
plt.grid(True)
plt.close('all')

# save results
stride = 100
with h5py.File('cdecay.h5', 'w') as fh5:
    fh5.create_dataset('absv', data=values_sorted['v'][::stride])
    fh5.create_dataset('csum', data=csum[::stride])
