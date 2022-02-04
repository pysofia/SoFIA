import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import minimize
import sofia.sampler as mcmc
import sofia.distributions as dist
import sofia.data_assembly as assemble
import mutationpp as mpp
import seaborn as sns
import pickle
import json
import sys

## Setting up Mutation++ options and mixture
opts = mpp.MixtureOptions("nitrogen-ions-carbon_9_olynick99")
opts.setThermodynamicDatabase("RRHO")
opts.setStateModel("ChemNonEq1T")

mix = mpp.Mixture(opts)

##
# Loading models and data
assembly = assemble.assembly('./cases.json','./models/models.json',mix)
measurements = ["Ps","Pd","Tw","density"]
assembly.lik_hyperparams(measurements)
assembly.assembly_prior()

## Looking for the MAP point to start sampling ##
Xi = [0.5]*len(assembly.hyp)
res = scipy.optimize.minimize(assembly.m_log_likelihood,Xi,method='Nelder-Mead',tol=1e-6)
print("MAP found at: "+str(res.x))

# MCMC sampling
nburn = 10000
sampler = mcmc.metropolis(np.identity(len(assembly.hyp))*0.01,assembly.log_likelihood,nburn)

sampler.seed(res.x)
sampler.Burn()

nchain = 10000
XMCMC = np.zeros((nchain,len(assembly.hyp)))

for i in range(nchain):
    XMCMC[i] = sampler.DoStep(1)
    print("MCMC step: "+str(i))

# Saving the chain to external file
filename = './chain_'+sys.argv[1]+'_'+sys.argv[2]+'_'+sys.argv[3]+'.dat'

with open(filename,'w') as ch:
    for i in range(len(XMCMC)):
        for j in range(len(assembly.hyp)-1):
            ch.write(str(XMCMC[i,j])+' ')
        ch.write(str(XMCMC[i,len(assembly.hyp)-1])+'\n')

## Marginals plotting
Xi_denorm = np.zeros((nchain,len(assembly.hyp)))

for i in range(nchain):
    Xi_denorm[i] =  assembly.denormalization(XMCMC[i,:])

plt.figure()
sns.kdeplot(Xi_denorm[:,assembly.models[assembly.therm_model][assembly.model]["indexes"]["Gnit"]],label='gnit',shade=True)
# sns.kdeplot(Xi_denorm[:,model.models[sys.argv[3]][sys.argv[4]]["indexes"]["Grec"]],label='grec',shade=True)
plt.xlim(-4.,0.)
plt.legend()

plt.figure()
sns.kdeplot(Xi_denorm[:,assembly.models[assembly.therm_model][assembly.model]["indexes"]["Ps"]],label='Ps',shade=True)
plt.legend()

plt.figure()
sns.kdeplot(Xi_denorm[:,assembly.models[assembly.therm_model][assembly.model]["indexes"]["Pd"]],label='Pd',shade=True)
plt.legend()

plt.figure()
sns.kdeplot(Xi_denorm[:,assembly.models[assembly.therm_model][assembly.model]["indexes"]["Te"]],label='Te',shade=True)
plt.legend()
plt.show()

## Chain diagnostics
dict_var={0:'gnit'} # The way they are positioned in MCMC chain

var = [0] # Position in XMCMC chain

sampler_diag = mcmc.diagnostics(XMCMC,dict_var)
sampler_diag.chain_visual(1,var,1000)
sampler_diag.autocorr(100,1,var)