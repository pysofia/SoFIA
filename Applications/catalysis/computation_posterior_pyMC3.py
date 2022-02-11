import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import minimize
import sofia.distributions as dist
import mutationpp as mpp
import seaborn as sns
import pickle
import json
import sys
import pymc3 as pm
import theano
import theano.tensor as tt

# define a theano Op for our likelihood function
class LogLike(tt.Op):

    """
    Specify what type of object will be passed and returned to the Op when it is
    called. In our case we will be passing it a vector of values (the parameters
    that define our model) and returning a single "scalar" value (the
    log-likelihood)
    """
    itypes = [tt.dvector] # expects a vector of parameter values when called
    otypes = [tt.dscalar] # outputs a single scalar value (the log likelihood)

    def __init__(self, loglike):
        """
        Initialise the Op with various things that our log-likelihood function
        requires. Below are the things that are needed in this particular
        example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined
        data:
            The "observed" data that our log-likelihood function takes in
        x:
            The dependent variable (aka 'x') that our model requires
        sigma:
            The noise standard deviation that our function requires.
        """

        # add inputs as class attributes
        self.likelihood = loglike

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        theta, = inputs  # this will contain my variables

        # call the log-likelihood function
        logl = self.likelihood(theta)

        outputs[0][0] = np.array(logl) # output the log-likelihood

##
# Log-likelihood
log_lik = pickle.load(open("./models/MTAt4.sav", 'rb'))
def log_likelihood(x):
    return log_lik.mean(x)

def m_log_likelihood(x):
    return -1*log_likelihood(x)

## Looking for the MAP point to start sampling ##
# x = [0.5]*4
# res = scipy.optimize.minimize(m_log_likelihood,x,method='Nelder-Mead',tol=1e-6)
# print("MAP found at: "+str(res.x))

logl = LogLike(log_likelihood)

# MCMC sampling
names_coeffs = ["gamma_Qz","gamma_Cu","gamma_TPS"]
with pm.Model():

    gCu = pm.Uniform("gamma_Cu", lower=0., upper=1.) # Uniform priors with prescribed bounds. Look in the documentation of pyMC3 to find other prior models
    gQz = pm.Uniform("gamma_Qz", lower=0., upper=1.)
    gTPS = pm.Uniform("gamma_TPS", lower=0., upper=1.)

    # convert m and c to a tensor vector
    theta = tt.as_tensor_variable([gCu, gQz, gTPS])

    # use a DensityDist (use a lamdba function to "call" the Op)
    pm.DensityDist('likelihood', lambda v: logl(v), observed={'v': theta})

    step = pm.Metropolis()
    trace = pm.sample(10000, step=step,tune=5000, discard_tuned_samples=True, chains=2)

    # step = pm.NUTS() # NUTS MCMC sampler. You can also try Metropolis or any other in the pyMC3 library!\


pm.traceplot(trace, ["gamma_Qz","gamma_Cu","gamma_TPS"])
plt.show()