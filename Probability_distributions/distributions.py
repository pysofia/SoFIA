import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from scipy.stats import norm
import seaborn as sns

class Gaussian:

    def __init__(self,d,hyperparams):
        self.dim = d
        self.hyperparams = hyperparams #matrix or vector, depends on dimensions

        self.mu = [0.]*self.dim
        self.sigma = [0.]*self.dim

        for i in range(self.dim):
            self.mu[i] = self.hyperparams[i][0]
            self.sigma[i] = self.hyperparams[i][1]

    # PDF
    def get_one_pdf_value(self,x,pos): 
        return np.divide(1,np.sqrt(2*math.py)*self.sigma[pos])*np.exp(-np.power(np.abs(self.mu[pos] - x),2)/(2*np.power(self.sigma[pos],2)))

    def get_pdf_values(self,x,pos):
        values = [0.]*len(x)
        for i in range(len(x)):
            values[i] += self.get_one_pdf_value(x[i],pos)
        return values

    # CDF
    def get_one_cdf_value(self,x,pos): 
        return 0.5*(1+special.erf(np.divide(x-self.mu[pos],self.sigma[pos]*np.sqrt(2))))

    def get_cdf_values(self,x,pos):
        values = [0.]*len(x)
        for i in range(len(x)):
            values[i] += self.get_one_cdf_value(x[i],pos)
        return values

    # Inverse CDF
    def inv_cdf(self,x,pos):
        return self.sigma[pos]*np.sqrt(2)*special.erfinv((2*x-1))+self.mu[pos]

    # Generate random samples
    def get_one_sample(self,pos):
        u = np.random.random()
        return self.inv_cdf(u,pos)

    def get_samples(self,pos,nsamples):
        values = [0.]*nsamples
        for i in range(nsamples):
            values[i] = self.get_one_sample(pos)

        return values

class Uniform:

    def __init__(self,d,hyperparams):
        self.dim = d
        self.hyperparams = hyperparams #matrix or vector, depends on dimensions

        self.lb = [0.]*self.dim
        self.ub = [0.]*self.dim

        for i in range(self.dim):
            self.lb[i] = self.hyperparams[i][0]
            self.ub[i] = self.hyperparams[i][1]

        self.hypvol = np.subtract(self.ub,self.lb)

    def get_one_pdf_value(self,x,pos): 
        if x>=self.lb[pos] and x<=self.ub[pos]:
            return np.divide(1,np.prod(self.hypvol))
        else:
            return 0.

    def get_pdf_values(self,x,pos):
        values = [0.]*len(x)
        for i in range(len(x)):
            values[i] += self.get_one_pdf_value(x[i],pos)
        return values

# class LogUniform:

#     def __init__(self,d,hyperparams):
#         self.dim = d
#         self.hyperparams = hyperparams #matrix or vector, depends on dimensions

#         self.lb = [0.]*self.dim
#         self.ub = [0.]*self.dim

#         for i in range(self.dim):
#             self.lb[i] = np.log10(self.hyperparams[i][0])
#             self.ub[i] = np.log10(self.hyperparams[i][1])

#         self.hypvol = np.abs(np.subtract(self.ub,self.lb))

#     def get_one_pdf_value(self,x,pos): 
#         if np.log10(x)>=self.lb[pos] and np.log10(x)<=self.ub[pos]:
#             return np.divide(1,np.prod(self.hypvol))
#         else:
#             return 0.

#     def get_pdf_values(self,x,pos):
#         values = [0.]*len(x)
#         for i in range(len(x)):
#             values[i] = self.get_one_pdf_value(x[i],pos)
#         return values

hyp = [[8.,0.2]]
h = [[0.,5.]]

Ps = Gaussian(1,hyp)
P = Gaussian(1,h)

x = np.linspace(0.00001,10.,1000)
y = np.linspace(-8.,8.,1000)

plt.plot(y,Ps.get_cdf_values(y,0))
plt.plot(y,P.get_cdf_values(y,0))
# plt.xscale('log')
plt.show()

sns.kdeplot(Ps.get_samples(0,10000))
plt.show()