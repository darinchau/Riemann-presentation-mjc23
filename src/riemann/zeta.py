from riemann.functions import ComplexFunction
import numpy as np
import scipy.special as sp

# Returns true if there is any nan
def has_nan(array):
    if np.isnan(array.real).any():
        return True
    if np.isnan(array.imag).any():
        return True
    return False

# identity is actually used to clean up stuff
def id(z):
    z[z.real == np.nan] = np.infty
    z[z.imag == np.nan] = np.infty
    return z

# By default we treat all nan as infinity
def inv(z):
    res = np.zeros_like(z)
    res[z != 0] = 1 / z[z != 0]
    res[z == 0] = np.infty
    return id(res)

def pow(a, b):
    # Returns a ** b with usual broadcasting rules
    return np.exp(b * np.log(a))

# The riemann zeta function with analytic continuation
class Zeta(ComplexFunction):
    def __init__(self):
        self.precision = 1e-12

    def set_precision(self, p):
        self.precision = p

    def call(self, z):
        # If the number is this close to 1 then return infinity
        pole_dist = 1e-5
        z = np.array(z)
        result_ = np.zeros_like(z)
        # Handle separately z = 0 (to avoid a 0 ^ 0 later) and poles (for obvious reasons)
        result_[np.abs(z-1) < pole_dist] = np.infty
        result_[np.abs(z-1) >= pole_dist] = self.riemann_siegel(z[np.abs(z-1) >= pole_dist])
        return result_

    # Epsilon is the convergence criterion
    # Implemented with reference to https://codegolf.stackexchange.com/questions/75277/evaluate-the-riemann-zeta-function-at-a-complex-number
    # and equation 20 in https://mathworld.wolfram.com/RiemannZetaFunction.html
    def riemann_siegel(self, z):
        # Create copy of array, and handle sccalars simultaneously (scalars become shape () array)
        result_ = np.ones_like(z, dtype = np.complex128) / 2

        # Precompute negative z with a new axis since we only need it once
        neg_z = -np.array(z[:, np.newaxis])

        # Precompute first factor since we can just get the first n
        neg_1_to_k = np.ones((1001,))
        neg_1_to_k[1::2] = -1

        # The summation part. At most 1000 iterations or break if the sum appears to converge
        # First iteration is always 1/2 and it causes some trouble anyway so we have precomputed it in result_
        for n in range(1, 1000):
            # Calculate (-1)^k * C(n, k) * (k+1)^(-s), which we will abbreviate as a, b, c
            # We handle the summation all at once using numpy magic,
            # keep in mind the summation goes from 0 to n, so there is n+1 iterations
            a = neg_1_to_k[:n+1]

            # Here we use an iterative approach to generate all binomial coeffs
            b = np.ones((n+1,))
            binomial_coeff = 1
            for k in range(n + 1):
                b[k] = binomial_coeff
                binomial_coeff = binomial_coeff * (n-k) / (k+1)

            # Use exp and log because now the usual broadcasting rules apply
            # This logarithm should be well defined because k's are all positive integers
            c = pow(np.arange(1, n+2), neg_z)


            # Compute the product (summand inside big summation)
            product = np.product(a*b*c, axis=-1) * (2**(-n-1))

            print(f"Max conv = {np.max(np.abs(product))}")

            result_ += product

            if np.all(np.abs(product) < self.precision):
                break

        factor = inv(1 - pow(2, 1-z))

        return factor * result_


zeta = Zeta()

# Import antics
__all__ = [
    "zeta"
]

# Test
if __name__ == "__main__":
    zeta.plot((-10, 10, -10, 10), dpi=15)
