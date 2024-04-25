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

# Vectorized standard normal PDF function
def stdNormal1(x):
    mean = 0
    std_dev = 1
    pdf = (1 / (std_dev * math.sqrt(2 * math.pi))) * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)
    return pdf


# Vectorized standard normal PDF function with mean -3
def stdNormal2(x):
    mean = -3
    std_dev = 1
    pdf = (1 / (std_dev * math.sqrt(2 * math.pi))) * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)
    return pdf

# Vectorized standard normal PDF function with mean 3 and standard deviation 3
def stdNormal3(x):
    mean = 3
    std_dev = 3
    pdf = (1 / (std_dev * math.sqrt(2 * math.pi))) * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)
    return pdf

def weighted_normal(x):
    return 0.7*stdNormal2(x) + 0.3*stdNormal3(x)

def func2(x):
    return scipy.special.eval_legendre(i,x )**2

#The function itself
def rosenbrock_function(x1, x2):
    minus_lnL = np.array(np.power((1.-x1), 2) + 100.* np.power((x2-x1**2),2),dtype=float)
    return minus_lnL


# One dimensional example
def integrate_me(f, distrib, npts):
    x = distrib.rvs(npts)
    ps = distrib.pdf(x)
    f_values = f(x)
    mu = np.mean(f_values / ps)
    err = np.std(f_values / ps) / np.sqrt(npts)
    return mu, err


print("(1)Integral estimate for f(x) = 1 on [10, 10]:" , integrate_me(func1, scipy.stats.uniform(loc=-10, scale=20), 1000))

for j in range(0, 6):
    print("(2, i = " + str(i) + " )The caculated mean and error for Legendre polynomial P{}(x) is: " , integrate_me(func2, scipy.stats.uniform(loc=-1, scale=2), 1000))
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

# Define truncated standard normal distribution
trunc_std_normal = scipy.stats.truncnorm(-3, 3, loc=0, scale=1)

# Function to integrate using truncated standard normal distribution
def integrate_with_truncated_normal(f, npts):
    x = trunc_std_normal.rvs(npts)
    ps = trunc_std_normal.pdf(x)
    f_values = f(x)
    mu = np.mean(f_values / ps)
    err = np.std(f_values / ps) / np.sqrt(npts)
    return mu, err

# Repeat the calculations for parts (a), (c), and (d) using truncated standard normal distribution
print("(2a) Integration on [10, 10] with f(x) = 1: ", integrate_with_truncated_normal(lambda x: 1, 1000))
print("(2c) Integration of standard normal PDF (truncated):", integrate_with_truncated_normal(stdNormal1, 1000))
print("(2d) Integration of weighted normal distribution: ", integrate_with_truncated_normal(weighted_normal, 1000))

# Example of integration
my2d = MyTwoDUniform(bounds=[[-5,5],[-5,5]])
mu = np.array([1, -0.5])
cov = np.array([[1., -0.1], [-0.1, 0.05]])

print("Main Project:", integrate_me(lambda x: rosenbrock_function(*(x.T)), my2d, 1000))
