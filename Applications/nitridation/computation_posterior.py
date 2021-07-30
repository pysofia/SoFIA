import numpy as np
import scipy
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, RBF
from scipy.optimize import minimize
import sofia.sampler as mcmc
import seaborn as sns
import matplotlib.pyplot as plt
import mutationpp as mpp
import sofia.distributions as dist

## Setting up Mutation++ options and mixture
opts = mpp.MixtureOptions("air_11")
opts.setThermodynamicDatabase("RRHO")
opts.setStateModel("ChemNonEq1T")

mix = mpp.Mixture("air_11")
mix = mpp.Mixture(opts)

##

def ve_betae(X): # X = (T, pres, Pd)
    
    mix.equilibrate(X[0], X[1])
    rhoe = mix.density()

    Vs_ext = np.power(2*X[2]/1.1/rhoe,0.5) #; //Kp = 1.1
    
    ve = Vs_ext*0.358134725

    V_torch = ve/0.3000427364 #; //NDP_ve = NDP4
    betae = V_torch*0.5169989517/0.025

    vect = np.zeros(2)
    vect[0] += ve
    vect[1] += betae

    return vect

## Priors ##
hyp = [[-4.,0.],[-4.,0.],[1200.,1700.],[2000.,4000.],[200,360.],[9000.,13000.]] # [Gnit,Grec,Ps,Tw,Pd,Te]
prior = dist.Uniform(6,hyp)

##

# Prior denormalization
def denormalization(Xi,hyp):
    for i in range(len(hyp)):
        Xi[i] = prior.lb[i]+(Xi[i]*(prior.ub[i]-prior.lb[i]))
    return Xi

## GP models ##
def rho(V):
    return GP_rho.predict(V)

def rec(V):
    return GP_rec.predict(V)

##

## Likelihood definition
h=[[1500.,22.5],[2410.,12.05],[268.,2.68],[1.64e-06,0.275e-06],[0.8e-06,1.0e-07]] # [Ps,Tw,Pd,rec,rho]
Lik = dist.Gaussian(5,h)

## Log-likelihood function
def log_likelihood(Xi):

    for i in range(len(Xi)):
        if Xi[i]<0:
            return -1.e16
        elif Xi[i]>1:
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

    value = 0.
    for i,j in zip(range(3),range(3)):
        value += np.log(Lik.get_one_pdf_value(prior.lb[j+2]+(Xi[j+2]*(prior.ub[j+2]-prior.lb[j+2])),i)+1.e-16)

    value += np.log(Lik.get_one_pdf_value(np.power(10,rec(V)),3)+1.e-16)+np.log(Lik.get_one_pdf_value(np.power(10,rho(V)),4)+1.e-16)

    return value

# def m_log_likelihood(Xi):
#     return -1*log_likelihood(Xi)

## log Likelihood to sample from ##
# def log_Lik(Xi):

#     for i in range(len(Xi)):
#         if Xi[i]<0:
#             return -1.e16
#         elif Xi[i]>1:
#             return -1.e16

#     V = np.zeros(7)
#     V[0] += Xi[0] # //Ps
#     V[1] += Xi[1] # //Tw
#     V[2] += Xi[2] # //Te
#     V[5] += Xi[4] # //Gnit
#     V[6] += Xi[5] # //Grec

#     X = np.zeros(3)
#     X[0] += Te_value(Xi[2])
#     X[1] += Ps_value(Xi[0])
#     X[2] += Pd_value(Xi[3])

#     ve, betae = ve_betae(X)

#     V[3] += (ve - 300.)/900. # //ve
#     V[4] += (betae - 20000.)/44000. # //beta e
#     V = [V]
 
#     return -np.power(np.abs(Ps_value(0.6)-Ps_value(Xi[0])),2)/(2*np.power(22.5,2))-np.power(np.abs(0.8e-06-np.power(10,rho(V))),2)/(2*np.power(1.0e-07,2))-np.power(np.abs(Tw_value(0.205)-Tw_value(Xi[1])),2)/(2*np.power(12.05,2))-np.power(np.abs(Pd_value(0.425)-Pd_value(Xi[3])),2)/(2*np.power(2.68,2))-np.power(np.abs(1.64e-06-np.power(10,rec(V))),2)/(2*np.power(0.275e-06,2))

## - log lik to minimize by Nelder-Mead to find MAP ##
# def Lik(Xi):

#     for i in range(len(Xi)):
#         if Xi[i]<0:
#             return -1.e16
#         elif Xi[i]>1:
#             return -1.e16

#     V = np.zeros(7)
#     V[0] += Xi[0] # //Ps
#     V[1] += Xi[1] # //Tw
#     V[2] += Xi[2] # //Te
#     V[5] += Xi[4] # //Gnit
#     V[6] += Xi[5] # //Grec

#     X = np.zeros(3)
#     X[0] += Te_value(Xi[2])
#     X[1] += Ps_value(Xi[0])
#     X[2] += Pd_value(Xi[3])

#     ve, betae = ve_betae(X)

#     V[3] += (ve - 300.)/900. # //ve
#     V[4] += (betae - 20000.)/44000. # //beta e
#     V = [V]
 
#     return -1*(-np.power(np.abs(Ps_value(0.6)-Ps_value(Xi[0])),2)/(2*np.power(22.5,2))-np.power(np.abs(0.8e-06-np.power(10,rho(V))),2)/(2*np.power(1.0e-07,2))-np.power(np.abs(Tw_value(0.205)-Tw_value(Xi[1])),2)/(2*np.power(12.05,2))-np.power(np.abs(Pd_value(0.425)-Pd_value(Xi[3])),2)/(2*np.power(2.68,2))-np.power(np.abs(1.64e-06-np.power(10,rec(V))),2)/(2*np.power(0.275e-06,2)))
##
## Loading training data ##
realizations_dir = '/Users/anabel/Documents/PhD/Code/stagline/Nitridation_1T/Recombination/realizations_G5_incomplete.dat'
data_dir = '/Users/anabel/Documents/PhD/Code/stagline/Nitridation_1T/Recombination/points_incomplete.dat'

X = np.loadtxt(data_dir)
Y = np.loadtxt(realizations_dir)

## Training GPs ##
kernel = Matern(length_scale=2, nu=3/2)
# kernel = RBF(length_scale=2)

GP_rec = gaussian_process.GaussianProcessRegressor(kernel=kernel,normalize_y=False)
GP_rec.fit(X, Y[:,0])

GP_rho = gaussian_process.GaussianProcessRegressor(kernel=kernel,normalize_y=False)
GP_rho.fit(X, Y[:,1])

## Looking for the MAP point to start sampling ##
Xi = [0.5]*6
# res = scipy.optimize.minimize(m_log_likelihood,Xi,method='Nelder-Mead',tol=1e-6)
# print("MAP found at: "+str(res.x))

# MCMC sampling
sampler = mcmc.metropolis(np.identity(6)*0.01,log_likelihood,100000)

par = Xi
sampler.seed(par)
sampler.Burn()

XMCMC = []

nchain = 100000
for i in range(nchain):
    XMCMC.append(sampler.DoStep(1))
    print('Step: '+ str(i))

XMCMC = np.array(XMCMC)

with open('./chain_SoFIA.dat','w') as ch:
    for i in range(len(XMCMC)):
        ch.write(str(XMCMC[i,0])+' '+str(XMCMC[i,1])+' '+str(XMCMC[i,2])+' '+str(XMCMC[i,3])+' '+str(XMCMC[i,4])+' '+str(XMCMC[i,5])+'\n')

## Trace plotting

sns.kdeplot(4*XMCMC[:,-2]-4.,label='gnit',shade=True)
plt.xlim(-4.,0.)
# sns.kdeplot(XMCMC[:,-1],label='b')
plt.legend()
plt.show()

## Chain diagnostics

dict_var={0:'a',-1:'b'} # The way they are positioned in MCMC chain

var = [-1] # Position in XMCMC chain

sampler_diag = mcmc.diagnostics(XMCMC,dict_var)
sampler_diag.chain_visual(1,var)
sampler_diag.autocorr(100,1,var)