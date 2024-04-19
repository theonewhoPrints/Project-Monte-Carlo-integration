import numpy as np
import scipy.stats

"""
file: MonteCarlo.py
course: MATH 251 
Final Project: MonteCarloIntegration
authors: Isaac Soares, Kaden Shuman, Chang you Yu

Purpose: maths

"""
npts = int

distrib0 = scipy.stats.truncnorm(-3,3,loc=0,scale=1)
distrib1 = scipy.stats.uniform(loc=-3,scale=6)

# One dimensional example
def integrate_me(f, distrib, npts=100):
    x = distrib.rvs(npts)
    ps = distrib.pdf(x)
    f = f(x)
    mu = np.mean(f/ps)
    err = np.std(f/ps)/np.sqrt(npts)
    return mu,err

# def integrate_me(f, distrib, npts=100):