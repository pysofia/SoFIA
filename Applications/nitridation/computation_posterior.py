import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import minimize
import sofia.sampler as mcmc
import sofia.distributions as dist
import mutationpp as mpp
import seaborn as sns
import pickle
import json
import sys

## Setting up Mutation++ options and mixture
opts = mpp.MixtureOptions("nitrogen-ions-carbon_9_olynick99")
opts.setThermodynamicDatabase("RRHO")
opts.setStateModel("ChemNonEq1T")

mix = mpp.Mixture("nitrogen-ions-carbon_9_olynick99")
mix = mpp.Mixture(opts)

##

# Loading stagline models
with open('./models/models.json') as json_file:
    models = json.load(json_file)

rec_mod = pickle.load(open(models[sys.argv[3]][sys.argv[4]][0], 'rb'))
rho_mod = pickle.load(open(models[sys.argv[3]][sys.argv[4]][1], 'rb'))

##

def ve_betae(X): # X = (T, pres, Pd)
    
    mix.equilibrate(X[2], X[1])
    rhoe = mix.density()

    Vs_ext = np.power(2*X[0]/1.1/rhoe,0.5) #; //Kp = 1.1
    
    ve = Vs_ext*0.358134725

    V_torch = ve/cases[sys.argv[1]]["NDPs"][3] #; //NDP_ve = NDP4
    betae = V_torch*0.5169989517/0.025

    vect = np.zeros(2)
    vect[0] += ve
    vect[1] += betae

    return vect

## GP models ##
def rho(V):
    return rho_mod.predict(V)[0]

def rec(V):
    return rec_mod.predict(V)[0]

##
## Priors ##
hyp = [[-4.,0.],[-4.,0.],[1200.,1700.],[2000.,4000.],[200,360.],[9000.,13000.]] # [Gnit,Grec,Ps,Tw,Pd,Te]
prior = dist.Uniform(len(hyp),hyp)

##
# Prior denormalization
def denormalization(Xi,hyp):
    Xi_denorm = [0.]*len(Xi)
    for i in range(len(hyp)):
        Xi_denorm[i] = prior.lb[i]+(Xi[i]*(prior.ub[i]-prior.lb[i]))
    return Xi_denorm

## Likelihood definition: setting up the experimental case to run
with open('cases.json') as json_file:
    cases = json.load(json_file)

if sys.argv[2] =='all':
    h = [[1500.,22.5],[cases[sys.argv[1]]['Tw']['mean'],cases[sys.argv[1]]['Tw']['std-dev']],[cases[sys.argv[1]]['Pd']['mean'],cases[sys.argv[1]]['Pd']['std-dev']],[cases[sys.argv[1]]['recession']['mean'],cases[sys.argv[1]]['recession']['std-dev']],[cases[sys.argv[1]]['density']['mean'],cases[sys.argv[1]]['density']['std-dev']]]
else:
    h = [[1500.,22.5],[cases[sys.argv[1]]['Tw']['mean'],cases[sys.argv[1]]['Tw']['std-dev']],[cases[sys.argv[1]]['Pd']['mean'],cases[sys.argv[1]]['Pd']['std-dev']],[cases[sys.argv[1]][sys.argv[2]]['mean'],cases[sys.argv[1]][sys.argv[2]]['std-dev']]]

Lik = dist.Gaussian(len(h),h)

## Log-likelihood function
def log_likelihood(Xi):

    for i in range(len(Xi)):
        if Xi[i]<0 or Xi[i]>1:
            return -1.e16

    V = np.zeros(7)
    V[0] += Xi[2] # //Ps
    V[1] += Xi[3] # //Tw
    V[2] += Xi[5] # //Te
    V[5] += Xi[0] # //Gnit
    V[6] += Xi[1] # //Grec

    X = np.zeros(3)
    X[0] += prior.lb[4]+(Xi[4]*(prior.ub[4]-prior.lb[4]))
    X[1] += prior.lb[2]+(Xi[2]*(prior.ub[2]-prior.lb[2]))
    X[2] += prior.lb[5]+(Xi[5]*(prior.ub[5]-prior.lb[5]))

    ve, betae = ve_betae(X)

    V[3] += (ve - 300.)/900. # //ve
    V[4] += (betae - 20000.)/44000. # //beta e
    V = [V]

    Xi_denorm = denormalization(Xi,hyp)

    return Lik.get_one_prop_logpdf_value(Xi_denorm[2],0) + Lik.get_one_prop_logpdf_value(Xi_denorm[3],1) + Lik.get_one_prop_logpdf_value(Xi_denorm[4],2) + Lik.get_one_prop_logpdf_value(np.power(10,rho(V)),4) + Lik.get_one_prop_logpdf_value(np.power(10,rec(V)),3)

def m_log_likelihood(Xi):
    return -1*log_likelihood(Xi)

## Looking for the MAP point to start sampling ##
Xi = [0.5]*6
res = scipy.optimize.minimize(m_log_likelihood,Xi,method='Nelder-Mead',tol=1e-6)
print("MAP found at: "+str(res.x))

# MCMC sampling
nburn = 100000
sampler = mcmc.metropolis(np.identity(len(hyp))*0.01,log_likelihood,nburn)

sampler.seed(res.x)
sampler.Burn()

nchain = 100000
XMCMC = np.zeros((nchain,len(hyp)))

for i in range(nchain):
    XMCMC[i] = sampler.DoStep(1)
    print("MCMC step: "+str(i))

with open('./chain_SoFIA.dat','w') as ch:
    for i in range(len(XMCMC)):
        ch.write(str(XMCMC[i,0])+' '+str(XMCMC[i,1])+' '+str(XMCMC[i,2])+' '+str(XMCMC[i,3])+' '+str(XMCMC[i,4])+' '+str(XMCMC[i,5])+'\n')

## Trace plotting

sns.kdeplot(4*XMCMC[:,0]-4.,label='gnit',shade=True)
sns.kdeplot(4*XMCMC[:,1]-4.,label='grec',shade=True)
plt.xlim(-4.,0.)
plt.legend()
plt.show()

## Chain diagnostics

dict_var={0:'gnit',1:'grec'} # The way they are positioned in MCMC chain

var = [0,1] # Position in XMCMC chain

sampler_diag = mcmc.diagnostics(XMCMC,dict_var)
sampler_diag.chain_visual(2,var,1000)
sampler_diag.autocorr(100,2,var)