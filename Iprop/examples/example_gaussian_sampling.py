import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sofia.sampler as mcmc

## 2D Gaussian
def func(par):
    for i in range(len(par)):
        if par[i]<-10.:
            return -np.power(np.abs(par[0] - 0.),2)/(2*np.power(1.,2))-np.power(np.abs(par[1] - 5.),2)/(2*np.power(1.,2))-1.e16
        elif par[i]>10.:
            return -np.power(np.abs(par[0] - 0.),2)/(2*np.power(1.,2))-np.power(np.abs(par[1] - 5.),2)/(2*np.power(1.,2))-1.e16
    return -np.power(np.abs(par[0] - 0.),2)/(2*np.power(1.,2))-np.power(np.abs(par[1] - 5.),2)/(2*np.power(1.,2))

## 1D Gaussian
def func1D(par):
    return -np.power(np.abs(par[0] - 0.),2)/(2*np.power(1.,2))

d=2
nburn = 10000
nchain = 10000
method = 'hamiltonian'

# MCMC sampling
sampler_choices = {'metropolis': mcmc.metropolis(np.identity(d),func,nburn), 'hamiltonian': mcmc.hamiltonian(func,d,path_len=0.5,step_size=0.1)}

sampler = sampler_choices[method]

par = [1.]*d
sampler.seed(par)
if method=='metropolis':
    sampler.Burn()

XMCMC = np.zeros((nchain,d))

for i in range(nchain):
    XMCMC[i] = sampler.DoStep(1)
    print("MCMC step: "+str(i))

## Trace plotting
true1 = [np.random.normal(loc=0.,scale=1.) for i in range(nchain)]
true2 = [np.random.normal(loc=5.,scale=1.) for i in range(nchain)]

sns.kdeplot(XMCMC[:,0],shade=True)
sns.kdeplot(true1,linestyle = '--',label='True')
plt.xlim(-5.,5.)
plt.legend()
plt.show()

sns.kdeplot(XMCMC[:,1],shade=True)
sns.kdeplot(true2,linestyle = '--',label='True')
plt.xlim(-2.,10.)
plt.legend()
plt.show()

plt.scatter(XMCMC[:,0],XMCMC[:,1])
plt.show()

## Chain diagnostics

dict_var={0:'$\\mu_{1}$', 1:'$\\mu_{2}$'} # The way they are positioned in MCMC chain

var = [0,1]

sampler_diag = mcmc.diagnostics(XMCMC,dict_var)
sampler_diag.chain_visual(2,var,5000)
sampler_diag.autocorr(70,2,var)