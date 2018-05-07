import re
from .libpyFFRT import Lambda, rt_type
import numpy as np

def get_basis(fname='rt_basis.desc'):
    """
    Keyword Arguments:
    fname -- descriptor file
    """
    rt_typem = {'S': rt_type.s, 'X': rt_type.x, 'Y': rt_type.y, 'D': rt_type.d}
    basis = []
    with open(fname, 'r') as f:
        lines = f.readlines()
        for line in lines:
            match = re.match('\(j=([0-9]+),\s*([a-zA-Z])\s*,\s*k=([-0-9]*)\) ([0-9\.]+) ([0-9\.]+)',
                             line)
            if match:
                j = int(match.group(1))
                t = match.group(2)
                k = int(match.group(3))
                ty = float(match.group(4))
                tx = float(match.group(5))

                l = Lambda(j, rt_typem[t], k)
                basis.append({'lambda': l,  't': np.array([ty, tx])})
            else:
                raise Exception('fail')

    return basis
