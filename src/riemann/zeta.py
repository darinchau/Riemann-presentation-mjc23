from riemann.functions import ComplexFunction
import numpy as np
import mpmath as mp
import numbers

import mpmath as mp

def honest_zeta(z):
    if abs(z - 1) < 1e-9:
        return np.infty
    return complex(mp.zeta(z))

# The riemann zeta function with analytic continuation
class Zeta(ComplexFunction):
    def __init__(self):
        self.z = np.vectorize(honest_zeta)

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
    zeta.plot((-20, 20, -20, 20))
