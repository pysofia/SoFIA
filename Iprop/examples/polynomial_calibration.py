import numpy as np
import scipy
from scipy.optimize import minimize
import sofia.sampler as mcmc
import seaborn as sns
import matplotlib.pyplot as plt
import sofia.distributions as dist

## True model ##

true_coeffs = {'a_0': 10., 'a_1': -2., 'a_2': 0., 'a_3': 0., 'a_4': 0.} # a_0 + a_1*x + a_2*x^2 + a_3*x^3 + a_4*x^4
# true_coeffs = {'a_0': 10., 'a_1': -2., 'a_2': 7.5, 'a_3': -3.3, 'a_4': -3.2}

def polynomial (a,x): # a it's a dictionary
    value = 0.
    for i in range(len(a)):
        value += a['a_'+str(i)]*np.power(x,i)

    return value

## Noisy data generation ##

n_obs = 30
sigma_obs = 0.1

# Generating the observations
x_obs = np.random.uniform(low=0., high=1., size=(n_obs))
observations = [polynomial(true_coeffs,x_obs[i]) + np.random.normal(loc=0.,scale=sigma_obs) for i in range(len(x_obs))]

######

## priors ##
h = [[-5.,20.]]*len(true_coeffs)
prior = dist.Uniform(len(true_coeffs),h)

# Prior denormalization: in case we want to sample normalized quantities. Not used in this example
# def denormalization(Xi,h):
#     for i in range(len(h)):
#         Xi[i] = prior.lb[i]+(Xi[i]*(prior.ub[i]-prior.lb[i]))
#     return Xi

## Likelihood definition
hyp = [[observations[i],sigma_obs] for i in range(n_obs)]
Lik = dist.Gaussian(n_obs,hyp)

## Log-likelihood function
def log_likelihood(Xi):

    for i in range(len(Xi)):
        if Xi[i]<prior.lb[i]:
            return -1.e16
        elif Xi[i]>prior.ub[i]:
            return -1.e16

    # Xi = denormalization(Xi,h)
    coeffs = {'a_0': Xi[0], 'a_1': Xi[1], 'a_2': Xi[2], 'a_3': Xi[3], 'a_4': Xi[4]}

    value = 0.
    for i in range(n_obs):
        value += Lik.get_one_prop_logpdf_value(polynomial(coeffs,x_obs[i]),i)

    return value

## mLog-likelihood function for MAP estimate
def mlog_likelihood(Xi):
    return -1.*log_likelihood(Xi)

## Looking for the MAP point to start sampling ##
Xi = [7.]*len(true_coeffs)
res = scipy.optimize.minimize(mlog_likelihood,Xi,method='Nelder-Mead',tol=1e-6)
print("MAP found at: "+str(res.x))

# MCMC sampling
sampler = mcmc.metropolis(np.identity(len(true_coeffs))*0.01,log_likelihood,100000)

sampler.seed(res.x)
sampler.Burn()

XMCMC = []

nchain = 100000
for i in range(nchain):
    XMCMC.append(sampler.DoStep(1))
    print('Step: '+ str(i))

XMCMC = np.array(XMCMC)

## Plotting noisy observations and calibrated polynomial model ##

x = np.linspace(0.,1.,100)

f_mean = [0.]*len(x)
f_u = [0.]*len(x)
f_l = [0.]*len(x)

for i in range(len(x)):
    f = [0.]*1000
    for j in range(1000):
        coefs = {'a_0': XMCMC[j,0], 'a_1': XMCMC[j,1], 'a_2': XMCMC[j,2], 'a_3': XMCMC[j,3], 'a_4': XMCMC[j,4]}
        f[j] = polynomial(coefs,x[i])
    f_mean[i] = np.mean(f)
    f_u[i] = np.percentile(f,97.5)
    f_l[i] = np.percentile(f,2.5)

plt.scatter(x_obs,observations,label='Observations')
plt.plot(x,f_mean,label='Mean')
plt.plot(x,polynomial(true_coeffs,x),color='red',label='True model')
plt.fill_between(x,f_l,f_u,color='blue',alpha=0.2,label='95% C. I.')
plt.plot(x,f_l,linestyle='--', color='blue')
plt.plot(x,f_u,linestyle='--', color='blue')
plt.xlim(0.,1.)
plt.legend()
plt.show()

## Kernel plotting

for i in range(len(true_coeffs)):
    sns.kdeplot(XMCMC[:,i],label='a_'+str(i),shade=True)

plt.legend()
plt.show()

## Chain diagnostics

dict_var={0:'a_0', 1:'a_1', 2:'a_2', 3:'a_3', 4:'a_4'} # The way they are positioned in MCMC chain

var = [0,1,2,3,4] # Position in XMCMC chain

sampler_diag = mcmc.diagnostics(XMCMC,dict_var)
sampler_diag.chain_visual(5,var,5000)
sampler_diag.autocorr(40,5,var)