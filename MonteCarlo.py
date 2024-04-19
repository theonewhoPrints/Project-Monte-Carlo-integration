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

class MyTwoDUniform(object):
    def __init__(self, bounds=None):
        self.bounds =np.array(bounds)
    def rvs(self,npts):
        my_out = np.empty( (len(self.bounds),npts))
        for dim in np.arange(len(self.bounds)):
            my_out[dim] = np.random.uniform(low=self.bounds[dim][0], high=self.bounds[dim][1], size=npts)
        return my_out.T
    def pdf(self,x):
        V = np.prod([self.bounds[:,1]- self.bounds[:,0]])
        return np.ones(x.shape[0])/V
    # Example of integration
    my2d = MyTwoDUniform(bounds=[-3,3])
    from scipy.stats import multivariate_normal
    mu = np.array([1,-0.5])
    cov = np.array([[ 1. , -0.1 ], [-0.1 , 0.05]])
    def f(x1, x2):
        x = np.array([x1, x2]).T
        return multivariate_normal.pdf(x, mu, cov)
        integrate_me(lambda x: f(*(x.T)), my2d)