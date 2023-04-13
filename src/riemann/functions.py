## Base class for complex functions

from __future__ import annotations
from abc import abstractmethod as virtual
from typing import Iterable, Callable, Generic, TypeVar
import numpy as np
from numpy.typing import NDArray
import cmath
import numbers
import cv2
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

pi = np.pi

def is_callable(f):
    return "__call__" in f.__dir__()

# Returns the "hue" of the complex number, which is the argument mod 2pi
# Change to degree and normalize to [0, 179]
def hue(z):
    return (np.angle(z, deg = True) + 180) *.5

# Returns the "saturation" of the complex number, which is 1 if the magnitude is smaller than sat_cut (=N)
# but N/x when magnitude >= N
# And then normalize to 0, 255
def saturation(z, sat_cut = 10):
    res = np.ones_like(z, dtype=np.float64)
    res[np.abs(z) >= sat_cut] = sat_cut/np.abs(z[np.abs(z) >= sat_cut])
    return res * 255

# Returns the "saturation" of the complex number, as defined by the function below
# And then normalize to 0, 255
def value(z):
    mag = np.abs(z)
    return np.sqrt(np.sqrt(1- 1/(1+mag*mag))) * 255

# A C -> C function
class ComplexFunction:
    # Implement this
    # Assumes that the function is pure
    def call(self, z: complex) -> complex:
        raise NotImplementedError

    def _set_f(self, _f):
        self._f = _f

    # Function calling. This is purely to overload the function signature
    def __call__(self, t: complex | Iterable[complex] | NDArray[np.complexfloating] | Expression | Callable[[complex], complex]):
        # Overload this thing if it is an expression
        if isinstance(t, Expression):
            return Expression(lambda x: self(t(x)), t._name)

        if hasattr(self, "_f"):
            fnobj = self._f #type:ignore
        else:
            fnobj = self.call

        if isinstance(t, numbers.Complex):
            return fnobj(t) #type: ignore

        # Numpy array
        if isinstance(t, np.ndarray):
            # Function must be vectorized for now otherwise return error
            return fnobj(t) # type:ignore

        # Iterables
        if isinstance(t, list):
            return [fnobj(x) for x in t]

        if isinstance(t, tuple):
            return tuple(fnobj(x) for x in t)

        raise TypeError("Type not supported")

    # Plots the function
    # Sat cut: These two numbers define the range of the magnitude of z that takes full satuarion
    # Too large and then it will start to turn white, too small it will become black
    def plot(self, plot_range: tuple = (-1, 1, -1, 1), dpi = None):
        # Readability
        x_min, x_max, y_min, y_max = plot_range

        if dpi is None:
            dpi = round(360/(x_max - x_min))

        # Calculate resolution
        resolution = (x_max - x_min) * dpi, (y_max - y_min) * dpi

        # -- Create the plot of f(x) = x first --
        x_plot = x_min + (x_max - x_min) / resolution[1] * np.arange(resolution[1])
        y_plot = (y_max + (y_min - y_max) / resolution[0] * np.arange(resolution[0]))
        y_plot = y_plot.reshape(-1, 1) * 1j
        z = x_plot + y_plot

        # Let numpy do it's thing
        result: NDArray[np.complex128] = self(z) # type: ignore

        # Screw this
        del x_plot
        del y_plot
        del z

        # -- Convert complex number to HSV --
        satuarion_cutoff = 1

        # Use broadcasting magic to convert the grid into the right shape
        result_ = np.zeros(result.shape + (3,))

        result_[:, : ,0] = hue(result)
        result_[:, :, 1] = saturation(result, sat_cut=satuarion_cutoff)
        result_[:, :, 2]  = value(result)

        # -- Show graphics using opencv and matplotlib --
        image = cv2.cvtColor(np.array(result_, dtype = np.uint8), cv2.COLOR_HSV2RGB)

        # Display the plot
        plt.figure()
        plt.imshow(image, extent=(x_min, x_max, y_min, y_max))
        plt.show()

def frange(start, stop, step):
    while start < stop:
        yield complex(start)
        start += step

# Overload the plot to make a line only
class RealToComplexFunction(ComplexFunction):
    def plot(self, plot_domain: tuple = (0, 10), plot_range: tuple = (-5, 5), dpi = 20):
        divs = dpi * (plot_domain[1] - plot_domain[0])
        result = self(np.linspace(plot_domain[0], plot_domain[1], divs, endpoint=True, dtype = np.complex128))

        # Create figure and axis objects
        fig, ax = plt.subplots()

        # Set the x and y limits
        ax.set_xlim(plot_range[0], plot_range[1])
        ax.set_ylim(plot_range[0], plot_range[1])

        # Set the origin at the center of the graph
        ax.spines['left'].set_position('zero')
        ax.spines['bottom'].set_position('zero')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ax.plot(result.real, result.imag)

        # Show the plot
        plt.show()

class Expression:
    def __init__(self, f, name: str):
        self._f = f
        self._name = name

    def __call__(self, x):
        return self._f(x)

    ## Add, mul, neg, and inv together give the function field we need
    def __add__(self, other):
        if isinstance(other, Expression):
            if not self._name == other._name:
                raise ValueError("Function cannot be added due to different variables")
            return Expression(lambda x: self(x) + other(x), self._name)

        return Expression(lambda x: self(x) + other, self._name)

    def __neg__(self):
        return Expression(lambda x: -self(x), self._name)

    def __mul__(self, other):
        if isinstance(other, Expression):
            if not self._name == other._name:
                raise ValueError("Function cannot be multiplied due to different variables")
            return Expression(lambda x: self(x) * other(x), self._name)
        return Expression(lambda x: self(x) * other, self._name)

    def inv(self):
        return Expression(lambda x: 1/self(x), self._name)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if isinstance(other, Expression):
            if not self._name == other._name:
                raise ValueError("Function cannot be divided due to different variables")
            return Expression(lambda x: self(x) * other.inv()(x), self._name)
        return Expression(lambda x: self(x) / other, self._name)

    def __rtruediv__(self, other):
        return self.inv() * other

def Function(expr: Expression):
    assert isinstance(expr, Expression)
    if expr._name == "z":
        f = ComplexFunction()
    elif expr._name == "t":
        f = RealToComplexFunction()
    else:
        raise ValueError("How did you define an expression on your own?")
    f._set_f(expr) #type:ignore
    return f

# z is a complex variable
# t is a real variable
z = Expression(lambda x: x, "z")
t = Expression(lambda x: x, "t")

# Other functions for fun
def expr(f):
    def hiya(z):
        if isinstance(z, Expression):
            return Expression(lambda x: f(z(x)), z._name)
        return f(z)
    return hiya

@expr
def sin(z):
    return np.sin(z)

@expr
def cos(z):
    return np.cos(z)

@expr
def tan(z):
    return np.cos(z)

@expr
def exp(z):
    return np.exp(z)

## Tests
class Add1(ComplexFunction):
    def call(self, z: complex):
        return z + 1

class NotVectorized(ComplexFunction):
    def call(self, z: complex):
        # Make sure the unfactorized function still works
        return max(abs(z), 1)

def test1():
    add1 = Add1()
    add1(1+2j)
    add1(3)
    add1(4.5)
    a = np.zeros((5, 5), dtype=np.complex128)
    add1(a)

def test2():
    a = np.zeros((5, 5), dtype=np.complex128)
    nv = NotVectorized()
    nv(a)

def test3():
    f = Function(t + 1)
    print(f(1))

if __name__ == "__main__":
    # f = Function(z)
    # f.plot() #type:ignore

    # g = Function(1/(z*z+1))
    # print(g(np.arange(10)))
    # g.plot((-2, 2, -2, 2))
    j = 1j
    f = Function(exp(j * t))
    print(f(2 * pi))
