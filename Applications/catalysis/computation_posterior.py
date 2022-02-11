import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import minimize
import sofia.sampler as mcmc
import sofia.distributions as dist
import sofia.data_assembly_cat as assemble
import mutationpp as mpp
import seaborn as sns
import pickle
import json
import sys

##
# Loading models and data
assembly = assemble.assembly('./models/models.json')
assembly.assembly_hyperparameters()

## Looking for the MAP point to start sampling ##
Xi = [0.5]*len(assembly.hyp)
res = scipy.optimize.minimize(assembly.m_log_likelihood,Xi,method='Nelder-Mead',tol=1e-6)
print("MAP found at: "+str(res.x))

# MCMC sampling
nburn = 10000
sampler = mcmc.metropolis(np.identity(len(assembly.hyp))*0.01,assembly.log_likelihood,nburn)

sampler.seed(res.x)
sampler.Burn()

nchain = 100000
XMCMC = np.zeros((nchain,len(assembly.hyp)))

for i in range(nchain):
    XMCMC[i] = sampler.DoStep(1)
    print("MCMC step: "+str(i))

# Saving the chain to external file
filename = './chain_'+sys.argv[1]+'.dat'

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
sns.kdeplot(Xi_denorm[:,assembly.models[assembly.case]["inputs"]["Gcu"]],label='Gcu',shade=True)
sns.kdeplot(Xi_denorm[:,assembly.models[assembly.case]["inputs"]["Gqz"]],label='Gqz',shade=True)
sns.kdeplot(Xi_denorm[:,assembly.models[assembly.case]["inputs"]["GTPS"]],label='GTPS',shade=True)
plt.xlim(-4.,0.)
# plt.ylim(0.,3.)
plt.legend()
plt.show()

## Chain diagnostics
dict_var={0:'GTPS'} # The way they are positioned in MCMC chain

var = [0] # Position in XMCMC chain

sampler_diag = mcmc.diagnostics(XMCMC,dict_var)
sampler_diag.chain_visual(1,var,1000)
sampler_diag.autocorr(100,1,var)