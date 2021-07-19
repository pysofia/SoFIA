import numpy as np
import matplotlib.pyplot as plt
from numpy import random
import seaborn as sns
import sofia.sampler as mcmc

# Data-generating function
def ishigami (x,y,z,a,b):
    return np.sin(x) + a*np.power(np.sin(y),2)+b*np.power(z,4)*np.sin(x)

# Likelihood
def log_lik(par):
    a = 8.
    b = 1.
    sigma = 0.2

    logl = -np.power(np.abs(ishigami(0,1,2,a,b)-ishigami(0,1,2,par[0],par[1])),2)/(2*np.power(sigma*ishigami(0,1,2,a,b),2))-np.power(np.abs(ishigami(0,2,5,a,b)-ishigami(0,2,5,par[0],par[1])),2)/(2*np.power(sigma*ishigami(0,2,5,a,b),2))-np.power(np.abs(ishigami(1,8,5,a,b)-ishigami(1,8,5,par[0],par[1])),2)/(2*np.power(sigma*ishigami(1,8,5,a,b),2))
    return logl

# MCMC sampling
sampler = mcmc.metropolis(np.identity(2)*0.01,log_lik,100000)

par = np.array([0.5,0.5])
sampler.seed(par)
sampler.Burn()

XMCMC = []

nchain = 100000
for i in range(nchain):
    XMCMC.append(sampler.DoStep(1))

XMCMC = np.array(XMCMC)

## Trace plotting

sns.kdeplot(XMCMC[:,0],label='a')
sns.kdeplot(XMCMC[:,1],label='b')
plt.legend()
plt.show()

## Chain diagnostics

dict_var={0:'a',1:'b'} # The way they are positioned in MCMC chain

var = [1]

sampler_diag = mcmc.diagnostics(XMCMC,dict_var)
sampler_diag.chain_visual(1,var)
sampler_diag.autocorr(70,1,var)