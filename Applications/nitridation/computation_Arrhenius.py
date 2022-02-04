import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import minimize
import sofia.sampler as mcmc
import sofia.distributions as dist
import sofia.models as mdls
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
data = mdls.data('./cases.json')
data.define_likelihood()

model = mdls.model(data.cases,'./models/models.json',mix,data.Lik)
model.define_prior()

## Looking for the MAP point to start sampling ##
Xi = [0.5]*len(model.hyp)
res = scipy.optimize.minimize(model.m_log_likelihood_Arrhenius,Xi,method='Nelder-Mead',tol=1e-6)
print("MAP found at: "+str(res.x))

# MCMC sampling
nburn = 10000
sampler = mcmc.metropolis(np.identity(len(model.hyp))*0.01,model.log_likelihood_Arrhenius,nburn)

sampler.seed(res.x)
sampler.Burn()

nchain = 10000
XMCMC = np.zeros((nchain,len(model.hyp)))

for i in range(nchain):
    XMCMC[i] = sampler.DoStep(1)
    print("MCMC step: "+str(i))

# Saving the chain to external file
filename = './chain_'+sys.argv[1]+'_'+sys.argv[2]+'_'+sys.argv[3]+sys.argv[4]+'.dat'

with open(filename,'w') as ch:
    for i in range(len(XMCMC)):
        for j in range(len(model.hyp)-1):
            ch.write(str(XMCMC[i,j])+' ')
        ch.write(str(XMCMC[i,len(model.hyp)-1])+'\n')

## Marginals plotting
Xi_denorm = np.zeros((nchain,len(model.hyp)))

for i in range(nchain):
    Xi_denorm[i] =  model.denormalization(XMCMC[i,:])

plt.figure()
# sns.kdeplot(Xi_denorm[:,model.models[sys.argv[3]][sys.argv[4]]["indexes"]["Gnit"]],label='gnit',shade=True)
sns.kdeplot(Xi_denorm[:,model.models[sys.argv[3]][sys.argv[4]]["indexes"]["Grec"]],label='grec',shade=True)
plt.xlim(-4.,0.)
plt.legend()

plt.figure()
sns.kdeplot(Xi_denorm[:,model.models[sys.argv[3]][sys.argv[4]]["indexes"]["Ps"]],label='Ps',shade=True)
plt.legend()

plt.figure()
sns.kdeplot(Xi_denorm[:,model.models[sys.argv[3]][sys.argv[4]]["indexes"]["Pd"]],label='Pd',shade=True)
plt.legend()

plt.figure()
sns.kdeplot(Xi_denorm[:,model.models[sys.argv[3]][sys.argv[4]]["indexes"]["Te"]],label='Te',shade=True)
plt.legend()
plt.show()

## Chain diagnostics
dict_var={0:'gnit'} # The way they are positioned in MCMC chain

var = [0] # Position in XMCMC chain

sampler_diag = mcmc.diagnostics(XMCMC,dict_var)
sampler_diag.chain_visual(1,var,1000)
sampler_diag.autocorr(100,1,var)