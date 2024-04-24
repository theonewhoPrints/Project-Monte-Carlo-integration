import numpy as np
import scipy.special
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

i = 0

def func1(x):
    return 1

def stdNormal1(x):
    mean = 0
    std_dev = 1
    pdf = (1 / (std_dev * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((x - mean) / std_dev) ** 2)
    return pdf

def stdNormal2(x):
    mean = -3
    std_dev = 1
    pdf = (1 / (std_dev * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((x - mean) / std_dev) ** 2)
    return pdf

def stdNormal3(x):
    mean = 3
    std_dev = 3
    pdf = (1 / (std_dev * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((x - mean) / std_dev) ** 2)
    return pdf

def weighted_normal(x):
    return 0.7*stdNormal2(x) + 0.3*stdNormal3(x)

def func2(x):
    return scipy.special.eval_legendre(i,x )

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

print("(1)The caculated mean and error is: " , integrate_me(func1, scipy.stats.uniform(loc=-10, scale=20), 1000))

for j in range(0, 6):
    print("(2, i = " + str(i) + " )The caculated mean and error is: " , integrate_me(func2, scipy.stats.uniform(loc=-1, scale=2), 1000))
    i+= 1

print("(3)Integration for standard normal " , integrate_me(stdNormal1, scipy.stats.uniform(loc=-10, scale=20), 1000))

print("(4)Integration for weighted normal " , integrate_me(weighted_normal, scipy.stats.uniform(loc=-10, scale=20), 1000))


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
my2d = MyTwoDUniform(bounds=[[-5,5],[-5,5]])
mu = np.array([1, -0.5])
cov = np.array([[1., -0.1], [-0.1, 0.05]])


def f(x1, x2):
    x = np.array([x1, x2]).T
    return multivariate_normal.pdf(x, mu, cov)


print("Main Project:", integrate_me(lambda x: f(*(x.T)), my2d, 1000))
