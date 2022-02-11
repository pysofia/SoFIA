import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.optimize import Bounds

class GP:

    def __init__(self,hyperparams):
        self.hyperparams = hyperparams

    def set_hyperparams(self,hyp):
        self.hyperparams = hyp

    def kernel(self,x,xp,hyp=None):
        if hyp is not None:
            self.set_hyperparams(hyp)

        return (self.hyperparams[0]**2)*np.exp(np.divide(-1*np.linalg.norm(x-xp)**self.hyperparams[2],self.hyperparams[2]*self.hyperparams[1]**2))

    def set_data(self,X,Y):
        self.np = len(Y)
        self.X = X
        self.Y = Y
        self.cov = np.ndarray((self.np,self.np))

    def k_star(self,x):
        k_st = np.array([0.]*self.np)
        for i in range(self.np):
            k_st[i] = self.kernel(x,self.X[i])
        return k_st

    def log_lik(self,hyp=None):
        if hyp is not None:
            self.set_hyperparams(hyp)

        for i in range(self.np):
            for j in range(self.np):
                self.cov[i,j] = self.kernel(self.X[i],self.X[j],hyp)

                if i==j:
                    self.cov[i,j] += self.hyperparams[3]**2

        self.cov_chol = np.linalg.cholesky(self.cov) # L

        beta = np.matmul(np.linalg.inv(self.cov_chol),self.Y)
        self.alpha = np.matmul(np.linalg.inv(np.transpose(self.cov_chol)),beta)

        chol_sum = 0.
        for i in range(self.np):
            chol_sum += np.log(self.cov_chol[i,i])

        return -0.5*np.matmul(np.transpose(self.Y),self.alpha) - chol_sum - (np.divide(self.np,2)*np.log(2*np.pi))

    def m_log_lik(self,hyp=None):
        return -1*self.log_lik(hyp)

    def train(self):
        hyp = self.hyperparams
        bounds = Bounds([1.e-3,1.e-4,1.1,1.e-5],[10.,1.e6,2.,5.])
        res = scipy.optimize.minimize(self.m_log_lik,hyp,method='trust-constr',bounds=bounds,tol=1e-6)
        self.hyperparams = res.x

    # def train(self):
    #     hyp = self.hyperparams
    #     res = scipy.optimize.minimize(self.m_log_lik,hyp,method='Nelder-Mead',tol=1e-6)
    #     self.hyperparams = res.x

    def mean(self,x):
        return np.matmul(np.transpose(self.k_star(x)),self.alpha)

    def variance(self,x):
        return self.kernel(x,x) - np.matmul(np.transpose(self.k_star(x)),np.matmul(np.linalg.inv(self.cov),self.k_star(x)))