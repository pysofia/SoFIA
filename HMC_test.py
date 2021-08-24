import numpy as np
import random
import scipy.stats as st
import matplotlib.pyplot as plt
import sofia.distributions as dist
import numdifftools as nd
import seaborn as sns

def normal(x,mu,sigma):
    numerator = np.exp(-1*((x-mu)**2)/(2*sigma**2))
    denominator = sigma * np.sqrt(2*np.pi)
    return numerator/denominator

def neg_log_prob(x,mu,sigma):
    return -1*np.log(normal(x=x,mu=mu,sigma=sigma))

def HMC(distr,path_len=1,step_size=0.25,initial_position=0.0,epochs=10_000):
    # setup
    steps = int(path_len/step_size) # path_len and step_size are tricky parameters to tune...
    samples = [initial_position]
    grad = nd.Gradient(distr.fun_logpdf())
    momentum_dist = dist.Gaussian(1,[[0,1]]) #st.norm(0, 1) 
    # generate samples
    for e in range(epochs):
        q0 = np.copy(samples[-1])
        q1 = np.copy(q0)
        p0 = np.random.normal() #momentum_dist.rvs()        
        p1 = np.copy(p0) 
        dVdQ = grad(q0) #-1*(q0-distr.mu[0])/(distr.sigma[0]**2) # gradient of PDF wrt position (q0) aka potential energy wrt position

        # leapfrog integration begin
        for s in range(steps): 
            p1 += step_size*dVdQ/2 # as potential energy increases, kinetic energy decreases, half-step
            q1 += step_size*p1 # position increases as function of momentum 
            p1 += step_size*dVdQ/2 # second half-step "leapfrog" update to momentum    
        # leapfrog integration end        
        p1 = -1*p1 #flip momentum for reversibility     

        
        #metropolis acceptance
        q0_nlp = -1*distr.get_one_prop_logpdf_value(q0) #+ np.log(distr.sigma[0] * np.sqrt(2*np.pi)) #neg_log_prob(x=q0,mu=distr.mu[0],sigma=distr.sigma[0])
        q1_nlp = -1*distr.get_one_prop_logpdf_value(q1) #+ np.log(distr.sigma[0] * np.sqrt(2*np.pi)) #neg_log_prob(x=q1,mu=distr.mu[0],sigma=distr.sigma[0])        

        p0_nlp = -1*momentum_dist.get_one_prop_logpdf_value(p0) #+ np.log(momentum_dist.sigma[0] * np.sqrt(2*np.pi)) #neg_log_prob(x=p0,mu=0,sigma=1)
        p1_nlp = -1*momentum_dist.get_one_prop_logpdf_value(p1) #+ np.log(momentum_dist.sigma[0] * np.sqrt(2*np.pi)) #neg_log_prob(x=p1,mu=0,sigma=1)
        
        # Account for negatives AND log(probabiltiies)...
        target = q0_nlp - q1_nlp # P(q1)/P(q0)
        adjustment = p1_nlp - p0_nlp # P(p1)/P(p0)
        acceptance = target + adjustment 
        
        event = np.log(random.uniform(0,1))
        if event <= acceptance:
            samples.append(q1)
        else:
            samples.append(q0)
    
    return samples
        
# mu = 1
# sigma = 3
hyp=[[0,1]]
distr = dist.Gaussian(1,hyp)
trial = HMC(distr,path_len=1.5,step_size=0.25)

lines = np.linspace(-6,6,10_000)
normal_curve = [normal(x=l,mu=distr.mu[0],sigma=distr.sigma[0]) for l in lines]

plt.plot(lines,normal_curve)
sns.kdeplot(trial,shade=True)
# plt.hist(trial,density=True,bins=20)
plt.show()