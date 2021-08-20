import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sofia.distributions as dist
import sofia.sampler as mcmc
from cycler import cycler
import mutationpp as mpp
import scipy
from scipy.optimize import minimize

# Plotting settings

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 18

plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
cc = cycler('linestyle', ['-', '--', ':', '-.', '-']) * cycler('color', ['r', 'g', 'b', 'm', 'k'])
plt.rc('axes', prop_cycle=cc)

plt.rc('text', usetex=True)
plt.rc('font', family='Helvetica')

##

# Time integration

def runge_kutta4(mix, rhoi, T, dt):
    
    mix.setState(rhoi,T,0)
    wdot_1 = mix.netProductionRates() 

    rhoi += 0.5 * np.multiply(dt, wdot_1)
    mix.setState(rhoi,T,0)
    wdot_2 = mix.netProductionRates() 
    
    rhoi += 0.5 * np.multiply(dt, wdot_2)
    mix.setState(rhoi,T,0)
    wdot_3 = mix.netProductionRates() 
    
    rhoi += np.multiply(dt,wdot_3)
    mix.setState(rhoi,T,0)
    wdot_4 = mix.netProductionRates() 

    return 1./6. * np.multiply(dt, (np.array(wdot_1) + 2 * np.array(wdot_2) + 2 * np.array(wdot_3) + np.array(wdot_4)))

def solver(params):

    # ---------------------------------
    # Write gsi.xml in mixture.xml files

    with open('/Users/anabel/Documents/PhD/Code/Mutationpp_Michele/Mutationpp/data/mechanisms/air5_Park.xml', 'r') as f:
        lines = f.readlines()

    # Replace the lines
    lines[12] = ('{:s}  \n').format('<arrhenius A="'+str(np.power(10,params[0]))+'" n="-1.6" T="113200.0" />')  
    lines[18] = ('{:s}  \n').format('<arrhenius A="'+str(np.power(10,params[1]))+'" n="-1.5" T="59360.0" />')  
    lines[24] = ('{:s}  \n').format('<arrhenius A="'+str(np.power(10,params[2]))+'" n="0.0" T="75500.0" />')  
    lines[30] = ('{:s}  \n').format('<arrhenius A="'+str(np.power(10,params[3]))+'" n="0.42" T="42938.0" />')  
    lines[35] = ('{:s}  \n').format('<arrhenius A="'+str(np.power(10,params[4]))+'" n="0.0" T="19400.0" />')  
    

    # Write the lines back
    with open('/Users/anabel/Documents/PhD/Code/Mutationpp_Michele/Mutationpp/data/mechanisms/air5_Park.xml', 'w') as f:
        f.writelines(lines)

    # ## Physico-chemical model settings (mpp)
    opts = mpp.MixtureOptions("air_5")
    opts.setThermodynamicDatabase("RRHO")
    opts.setStateModel("ChemNonEq1T")

    mix = mpp.Mixture(opts)
    ns = mix.nSpecies()
    nr = mix.nReactions()

    # ##

    T = 300
    P = 10000
    mix.equilibrate(T,P)
    rhoi_eq = mix.densities()

    Tinit = params[5]*1000. #15000
    mix.setState(rhoi_eq,Tinit,1)
    total_energy = mix.mixtureEnergyMass()*mix.density()

    time = 0.
    time_final = 1e-7
    dt = 1.e-9
    rhoi = rhoi_eq

    y0 = np.divide(rhoi_eq,mix.density())
    x0 = mix.convert_y_to_x(y0)
    temperature = np.array(mix.T())
    temp = [0]*1002
    # mass_fractions = np.array([y0])
    # mole_fraction =  np.array([x0])
    # time_reaction = np.array(time)

    while (time < time_final):
        dt = min(dt * 1.0002, 1.e-10)
        drhoi = runge_kutta4(mix, rhoi, total_energy, dt)
        rhoi += drhoi
        time += dt
        mix.setState(rhoi,total_energy,0)
        y = mix.Y()
        x = mix.convert_y_to_x(np.array(y))
        # mass_fractions = np.vstack((mass_fractions,[y]))
        # mole_fraction =  np.vstack((mole_fraction,[x]))
        temperature = np.vstack((temperature,mix.T()))
        # time_reaction  = np.vstack((time_reaction,time))

    for i in range(len(temp)):
        temp[i] += temperature[i][0]

    return temp

time_steps = 1002
time = [0]*time_steps
time_final = 1.e-7
for i in range(time_steps):
    time[i] += (i+1)*np.divide(time_final,time_steps)

## Prior distributions ##
hyp = [[20.,22.],[20.,22.],[10.,15.],[10.,15.],[10.,15.],[10.,15.]]
prior = dist.Uniform(6,hyp)

## Data consist of different temperature readings at different time steps ##

sigma = 100.
n_obs=1
true_params = [prior.get_one_sample(),prior.get_one_sample(pos=1),prior.get_one_sample(pos=2),prior.get_one_sample(pos=3),prior.get_one_sample(pos=4),prior.get_one_sample(pos=5)]
print(true_params)

output = solver(true_params)

t_obs = [1001] #[np.random.randint(0,1003) for i in range(n_obs)]
time_obs = [time[t_obs[i]] for i in range(n_obs)]

data = [output[t_obs[i]]+np.random.normal(loc=0,scale=sigma) for i in range(n_obs)] # Five random points in time to perform noisy measurement

## Likelihood definition
h = [[data[i],sigma] for i in range(n_obs)]
Lik = dist.Gaussian(n_obs,h)

## Log-likelihood function
def log_likelihood(Xi):

    for i in range(len(Xi)):
        if Xi[i]<prior.lb[i]:
            return -1.e16
        elif Xi[i]>prior.ub[i]:
            return -1.e16

    s = solver(Xi)

    value = 0.
    for i in range(n_obs):
        value += Lik.get_one_prop_logpdf_value(s[t_obs[i]],i)

    return value

## mLog-likelihood function for MAP estimate
def mlog_likelihood(Xi):
    return -1.*log_likelihood(Xi)

## Looking for the MAP point to start sampling ##
par = [21.,21.,11.,11.,11.,12.]
res = scipy.optimize.minimize(mlog_likelihood,par,method='Nelder-Mead',tol=1e-6)
print("MAP found at: "+str(res.x))

# MCMC sampling
sampler = mcmc.metropolis(np.identity(6)*0.01,log_likelihood,10000)

sampler.seed(res.x)
sampler.Burn()

XMCMC = []

nchain = 10000
for i in range(nchain):
    XMCMC.append(sampler.DoStep(1))
    print('Step: '+ str(i))

XMCMC = np.array(XMCMC)

## Plotting noisy observations and calibrated model ##

f_mean = [0.]*time_steps
f_u = [0.]*time_steps
f_l = [0.]*time_steps

sol = np.zeros((nchain,time_steps))

for i in range(nchain):
    sol[i] = solver(XMCMC[i])

for j in range(time_steps):
    f_mean[j] = np.mean(sol[:,j])
    f_u[j] = np.percentile(sol[:,j],97.5)
    f_l[j] = np.percentile(sol[:,j],2.5)

plt.scatter(time_obs,data,color='black',label='Observations')
plt.plot(time,f_mean,color='green',label='Mean')
plt.plot(time,output,color='red',label='True model')
plt.fill_between(time,f_l,f_u,color='green',alpha=0.2,label='95\% C. I.')
plt.plot(time,f_u,linestyle='--', color='green')
plt.plot(time,f_l,linestyle='--', color='green')
ax = plt.gca()
ax.yaxis.set_label_coords(0.0, 1.00)
plt.ylabel('Temperature, K', rotation=0)
plt.xlabel('time, s')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False) 
plt.xlim(0.,np.power(10.,-7.))
plt.legend() 
plt.show()

## Kernel plotting

dict_var = {0: 'N2+M=2N+M',1:'O2+M=2O+M',2:'NO+M=N+O+M',3:'N2+O=NO+N',4:'NO+O=O2+N',5:'$T_{\mathrm{initial}}$'} # The way they are positioned in MCMC chain

for i in range(5):
    sns.kdeplot(XMCMC[:,i],label=dict_var[i],shade=True)

plt.legend()
plt.show()

sns.kdeplot(XMCMC[:,5],label=dict_var[5],shade=True)

plt.legend()
plt.show()

## Chain diagnostics

var = [0,1,2,3,4,5] # Position in XMCMC chain

sampler_diag = mcmc.diagnostics(XMCMC,dict_var)
sampler_diag.chain_visual(6,var,5000) # 5 plots, show 5000 samples
sampler_diag.autocorr(100,6,var) # 70 lags