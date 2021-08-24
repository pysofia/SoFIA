import numpy as np
from numpy import random
from statsmodels.tsa.stattools import acf
import matplotlib.pyplot as plt
import sofia.distributions as dist
import numdifftools as nd

class metropolis:

    """

        This class creates a Metropolis-Hastings sampler (MCMC chain) with classical adaptation of the covariance matrix through a burn-in stage.

        INPUTS: Initial covariance matrix, the posterior (log-posterior, proportional), and number of samples to adapt the covariance matrix (burning stage)

        OUTPUTS: seed function to set some initial parameters of the chain, move one step by an accept-reject procedure, burning stage whose samples are used to adapt the covariance matrix. Other utilities are included to ease the use and bug finding.

    """

    def __init__(self, covinit, fun_in, nburn):
        self.cov = covinit
        self.fun = fun_in
        self.Tac = [0.]*nburn
        self.d = len(covinit)
        self.CovDev = np.linalg.cholesky(self.cov*(2.38*2.38)/self.d)
        # np.random.seed(2) # Comment or uncomment depending if you are trying to compare results and don't want stochastic noise.

    def seed(self,xini):
        self.xcur = xini
        self.fcur = self.fun(xini)
        self.mea = self.xcur
        self.acc = np.array([[0.]*len(self.xcur)]*len(self.xcur))
        for i in range(len(self.xcur)):
            for j in range(len(self.xcur)):
                self.acc[i][j] += (self.xcur[i]*self.xcur[j])
        self.count = 1

    def DoStep(self,nstep):
        for i in range(nstep):
            self.doOneStep()
        return self.xcur

    def doOneStep(self):
        xTemp = np.add(self.xcur, self.PropStep())
        fTemp = self.fun(xTemp)
        r0=np.random.random()
        if fTemp-self.fcur>np.log(r0):
            self.xcur = xTemp
            self.fcur = fTemp
            self.Tac[self.count%len(self.Tac)] = 1
        else:
            self.Tac[self.count%len(self.Tac)] = 0

    def Burn(self):
        print("Burning "+str(len(self.Tac))+" samples...")
        for i in range(len(self.Tac)):
            self.BurnOneStep()
            if i>300 and i%100==0: ## If iteration number is larger than 300 and a multiplo of 100
                self.AdaptCov()
        print("Accept rate " + str(np.sum(self.Tac)/len(self.Tac)) +"\n")

    def BurnOneStep(self):
        xTemp = np.add(self.xcur,self.PropStep())
        fTemp = self.fun(xTemp)
        r0 = np.random.random()
        if fTemp - self.fcur > np.log(r0):
            self.xcur = xTemp
            self.fcur = fTemp
            self.Tac[self.count%len(self.Tac)] = 1
        else:
            self.Tac[self.count%len(self.Tac)] = 0

        self.mea = np.add(self.mea,self.xcur)
        for i in range(len(self.xcur)):
            for j in range(len(self.xcur)):
                self.acc[i][j] += (self.xcur[i]*self.xcur[j])
        self.count += 1

    def GetCur(self):
        return self.xcur

    def SetCovProp(self,Cov_in):
        self.cov = Cov_in

    def GetCovProp(self):
        return self.cov

    def DecCov(self):
        self.CovDev = np.linalg.cholesky(self.cov*(2.38*2.38)/self.d)

    def PropStep(self):
        dX = [0.]*self.d
        for i in range(self.d):
            dX[i] = np.random.normal()
        dX = np.matmul(self.CovDev,dX)
        for i in range(self.d):
            dX[i] += np.random.normal()*1.e-6
        return dX

    def AdaptCov(self):
        c1 = self.count
        c2 = c1*c1
        self.cov = self.acc/(self.count-1)
        for i in range(len(self.xcur)):
            for j in range(len(self.xcur)):
                self.cov[i][j] -= (self.mea[i]*self.mea[j])/c2
        self.cov += 1.e-6 * np.identity(self.d)
        self.DecCov()

class diagnostics:

    """

        This class creates a diagnostics object by computing different widely accepted metrics for studying the convergence of the MCMC chain.

        INPUTS: MCMC chain (array of values) and a dictionary of variables. Each variable is attached to a number according to its position in the MCMC chain. The number can be associated to a name in the dictionary for plotting.

        OUTPUTS: Autocorrelation function, visualization of the chain.

    """

    def __init__(self,chain,dict_var):
        self.chain = chain
        self.dict_var = dict_var

    def autocorr(self,nlg,nplots,d):
        n = list(range(1, len(self.chain)+1))
        n_lg = [0.]*(1+nlg)
        for i in range(nlg+1): # nlg+1
            n_lg[i] = 1+i #1 + i*nlg
  
        for i in range(nplots):
            a = acf(self.chain[:,d[i]],nlags=nlg)
            plt.plot(n_lg,a,label=self.dict_var[d[i]])
            plt.fill_between(n_lg,0.,a,alpha=0.2)
        plt.xlim(0.,n_lg[-1])
        plt.ylabel('ACF')
        plt.legend()
        plt.show()

    def autocorr1d(self,nlg):
        n = list(range(1, len(self.chain)+1))
        n_lg = [0.]*(1+nlg)
        for i in range(nlg+1): # nlg+1
            n_lg[i] = 1+i #1 + i*nlg
  
        a = acf(self.chain[:],nlags=nlg)
        plt.plot(n_lg,a,label=self.dict_var[0])
        plt.fill_between(n_lg,0.,a,alpha=0.2)
        plt.xlim(0.,n_lg[-1])
        plt.ylabel('ACF')
        plt.legend()
        plt.show()

    def chain_visual(self,nplots,d,nsteps=100):
        n = list(range(1, len(self.chain)+1))
        for i in range(nplots):
            plt.plot(n[0:nsteps],self.chain[0:nsteps,d[i]],label=self.dict_var[d[i]])
        plt.xlim(0.,n[nsteps-1])
        plt.xlabel('chain steps')
        plt.legend()
        plt.show()

    def chain_visual1d(self,nsteps=100):
        n = list(range(1, len(self.chain)+1))
        plt.plot(n[0:nsteps],self.chain[0:nsteps],label=self.dict_var[0])
        plt.xlim(0.,n[nsteps-1])
        plt.xlabel('chain steps')
        plt.legend()
        plt.show()

class hamiltonian(dist.Gaussian):

    """

        This class creates a Hamiltonian Monte Carlo sampler (MCMC chain) with the identity matrix as the momentum distribution's covariance matrix.

        INPUTS: The posterior (log-posterior, proportional), dimensions (default 1), path length for the leapfrog integration (default 1), and step size along the path length for each leapfrog step (default 0.25).

        OUTPUTS: seed function to set some initial parameters of the chain, move one step by Hamiltonian dynamics, and an accept-reject procedure to counteract the effect of the numerics when solving for the Hamiltonian dynamics.

    """

    def __init__(self,fun_in,d=1,path_len=1,step_size=0.25):

        self.path_len = path_len
        self.step_size = step_size
        self.steps = int(self.path_len/self.step_size) # path_len and step_size are tricky parameters to tune...
        self.fun_in = fun_in
        self.momentum_dist = dist.Gaussian(d,[[0.,1.]]*d)
        self.grad = nd.Gradient(self.fun_in)
        self.d = d

    def seed(self,xini):
        self.xcur = xini

    def DoStep(self,nsteps):
        for i in range(nsteps):
            self.xcur=self.doOneStep()
        return self.xcur

    def doOneStep(self):
        q0 = self.xcur
        q1 = self.xcur
        p0=[0]*self.d
        for i in range(self.d):
            p0[i] = self.momentum_dist.get_one_sample(pos=i) ## Draw from a N(0.1) 
        p1 = p0 
        dVdQ = self.grad(q0) # gradient of PDF wrt position (q0) aka potential energy wrt position

        # leapfrog integration begin
        for s in range(self.steps): 
            p1 += self.step_size*dVdQ/2 # as potential energy increases, kinetic energy decreases, half-step
            q1 += self.step_size*p1 # position increases as function of momentum 
            p1 += self.step_size*dVdQ/2 # second half-step "leapfrog" update to momentum    
        # leapfrog integration end        
        p1 = -1*p1 #flip momentum for reversibility     

        # metropolis acceptance
        q0_nlp = -1*self.fun_in(q0)
        q1_nlp = -1*self.fun_in(q1) 

        p0_nlp = 0.
        p1_nlp = 0.
        for i in range(self.d):
            p0_nlp += -1*self.momentum_dist.get_one_prop_logpdf_value(p0[i],pos=i) # Only because momentum dist is taken with no correlations (identity covariance matrix)
            p1_nlp += -1*self.momentum_dist.get_one_prop_logpdf_value(p1[i],pos=i)     
        
        # Account for negatives AND log(probabilties)...
        target = q0_nlp - q1_nlp 
        adjustment = p1_nlp - p0_nlp
        acceptance = target + adjustment 
        
        event = np.log(np.random.uniform())
        if event <= acceptance:
            return q1
        else:
            return q0