import numpy as np
import math
import scipy.stats
from scipy.stats import multivariate_normal

"""
file: MonteCarlo.py
course: MATH 251 
Final Project: MonteCarloIntegration
authors: Isaac Soares, Kaden Shuman, Chang you Yu

Purpose: maths

"""
npts = int

distrib0 = scipy.stats.truncnorm(-10, 10, loc=0, scale=1)
distrib1 = scipy.stats.uniform(loc=-10, scale=20)

def func1(x):
    return 1

def stdNormal(x):
    pdf = (1 / (math.sqrt(2 * math.pi))) * math.exp(-0.5 * (x ** 2))
    return pdf

#The function itself
def rosenbrock_function(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2


# One dimensional example
def integrate_me(f, distrib, npts):
    x = distrib.rvs(npts)
    ps = distrib.pdf(x)
    sum = 0
    for val in x:
        sum += f(val)
    #f = f(x)
    mean = sum/npts
    f = mean
    mu = np.mean(f / ps)
    err = np.std(f / ps) / np.sqrt(npts)
    return mu, err


print("The caculated mean and error is: " , integrate_me(func1, distrib1, 100))

print("Integration for standard normal " , integrate_me(stdNormal, distrib1, 10000))

"""
class MyTwoDUniform(object):
    def __init__(self, bounds=None):
        self.bounds = np.array(bounds)

    def rvs(self, npts):
        my_out = np.empty((len(self.bounds), npts))
        for dim in np.arange(len(self.bounds)):
            my_out[dim] = np.random.uniform(low=self.bounds[dim][0], high=self.bounds[dim][1], size=npts)
        return my_out.T

    def pdf(self, x):
        V = np.prod([self.bounds[:, 1] - self.bounds[:, 0]])
        return np.ones(x.shape[0]) / V


# Example of integration
my2d = MyTwoDUniform(bounds=[-3, 3])
mu = np.array([1, -0.5])
cov = np.array([[1., -0.1], [-0.1, 0.05]])


def f(x1, x2):
    x = np.array([x1, x2]).T
    return multivariate_normal.pdf(x, mu, cov)


integrate_me(lambda x: f(*(x.T)), my2d)
"""