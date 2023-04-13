from riemann.functions import ComplexFunction
import numpy as np
from numpy.typing import NDArray
import mpmath as mp
import numbers
import h5py
from fractions import Fraction

from tqdm import tqdm
import mpmath as mp

import time

def honest_zeta(z):
    if abs(z - 1) < 1e-9:
        return np.infty
    return complex(mp.zeta(z))

# Ranges that interop with fractions
def frange(a: int, c: int, f: int):
    start_ = Fraction(a, 1)
    stop_ = Fraction(c, 1)
    step_ = Fraction(1, f)
    while start_ <= stop_:
        yield start_.numerator, start_.denominator
        start_ += step_

def create_data(groups):
    # Stores (a, b, c, d): zeta(a/b + c/d i)
    # a/b and c/d are in simplest form
    results: dict[tuple[int, int, int, int], complex] = {}

    for group in groups:
        for a, b, c, d in tqdm(group):
            z = (a/b) + (c/d) * 1j
            results[(a, b, c, d)] = complex(honest_zeta(z)) #type:ignore

    print(f"Created {len(results.keys())} datas")

    # Open an HDF5 file in write mode
    with h5py.File('zeta.h5', 'w') as f:
        group = f.create_group("values")
        for key, value in results.items():
            group.create_dataset(str(key), data=np.array([value.real, value.imag], dtype=np.float64), compression='gzip')

def gen_data():
    groups = [
        ((a, b, c, d) for a, b in frange(0, 10, 20) for c, d in frange(0, 10, 20)),
        ((a, b, c, d) for a, b in frange(0, 20, 10) for c, d in frange(0, 20, 10)),
        ((a, b, c, d) for a, b in frange(0, 50, 2) for c, d in frange(0, 50, 2)),
        ((1, 2, a, b) for a, b in frange(0, 100, 1))
    ]

    create_data(groups)

# Precompute some values cuz 🥱🥱🥱
zeta_h5 = h5py.File('zeta.h5', 'r')

results = {}
for k, z in zeta_h5["values"].items():
    key = tuple(int(a) for a in k[1:-1].split(","))
    results[key] = complex(z[0], z[1])

def zeta_with_query(z: complex):
    if z.imag < 0:
        return zeta_with_query(z.conjugate())

    re = Fraction(z.real).limit_denominator(100)
    im = Fraction(z.imag).limit_denominator(100)

    try:
        return results[(re.numerator, re.denominator, im.numerator, im.denominator)]
    except KeyError:
        return honest_zeta(z)

# The riemann zeta function with analytic continuation
class Zeta(ComplexFunction):
    def __init__(self):
        self.z = np.vectorize(zeta_with_query)

    def call(self, z):
        # the pole problem is handled in the query
        if isinstance(z, numbers.Complex):
            return self.z(z)
        return np.array(self.z(z), dtype=np.complex128)

zeta = Zeta()

# Import antics
__all__ = [
    "zeta"
]

# Test
if __name__ == "__main__":
    t = time.perf_counter()
    zeta.plot((-20, 20, -20, 20))
    t2 = time.perf_counter()
    print(t2 - t)
    # gen_data()
