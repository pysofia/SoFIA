import numpy as np
from scipy import special
from scipy.stats import norm

class Gaussian:

    """

        This class creates a Gaussian distribution object of uncorrelated variables in arbitrary dimension.

        INPUTS: Number of dimensions and hyperparameters: mean, variance. Along with instantiating a class object, we can compute PDF and CDF values and generate random samples

        OUTPUTS: A single value of the Gaussian PDF for each independent variable, an array of PDF values, a single CDF value, an array of CDF values, random samples.

    """

    def __init__(self,d,hyperparams):
        self.dim = d
        self.hyperparams = hyperparams #matrix or vector, depends on dimensions

        self.mu = [0.]*self.dim
        self.sigma = [0.]*self.dim

        for i in range(self.dim):
            self.mu[i] = self.hyperparams[i][0]
            self.sigma[i] = self.hyperparams[i][1]

    # PDF
    def get_one_pdf_value(self,x,pos=0): 
        return np.divide(1,np.sqrt(2*np.pi)*self.sigma[pos])*np.exp(-np.power(np.abs(self.mu[pos] - x),2)/(2*np.power(self.sigma[pos],2)))

    def get_one_prop_logpdf_value(self,x,pos=0): 
        return -np.power(np.abs(self.mu[pos] - x),2)/(2*np.power(self.sigma[pos],2))

    def get_pdf_values(self,x,pos=0):
        values = [0.]*len(x)
        for i in range(len(x)):
            values[i] += self.get_one_pdf_value(x[i],pos)
        return values

    # PDF as function
    def fun_pdf(self,pos=0):
        return lambda u: self.get_one_pdf_value(u,pos)

    # LogPDF as function
    def fun_logpdf(self,pos=0):
        return lambda u: self.get_one_prop_logpdf_value(u,pos)

    # CDF
    def get_one_cdf_value(self,x,pos=0): 
        return 0.5*(1+special.erf(np.divide(x-self.mu[pos],self.sigma[pos]*np.sqrt(2))))

    def get_cdf_values(self,x,pos=0):
        values = [0.]*len(x)
        for i in range(len(x)):
            values[i] += self.get_one_cdf_value(x[i],pos)
        return values

    # CDF as function
    def fun_cdf(self,pos=0):
        return lambda u: self.get_one_cdf_value(u,pos)

    # Evaluation of inverse CDF in x
    def inv_cdf(self,x,pos=0):
        return self.sigma[pos]*np.sqrt(2)*special.erfinv((2*x-1))+self.mu[pos]

    # Inverse CDF as function
    def fun_icdf(self,pos=0):
        return lambda u: self.inv_cdf(u,pos)

    # Generate random samples
    def get_one_sample(self,pos=0):
        u = np.random.random()
        return self.inv_cdf(u,pos)

    def get_samples(self,nsamples,pos=0):
        values = [0.]*nsamples
        for i in range(nsamples):
            values[i] = self.get_one_sample(pos)

        return values

class Uniform:

    """

        This class creates a Uniform distribution object of uncorrelated variables in arbitrary dimension.

        INPUTS: Number of dimensions and hyperparameters: lower and upper bounds. Along with instantiating a class object, we can compute PDF and CDF values and generate random samples

        OUTPUTS: A single value of the uniform PDF for each independent variable, an array of PDF values, a single CDF value, an array of CDF values, random samples.

        REMARK: This class can also be used with logarithmic values to compute the same properties for a Log-Uniform distribution

    """

    def __init__(self,d,hyperparams):
        self.dim = d
        self.hyperparams = hyperparams #matrix or vector, depends on dimensions

        self.lb = [0.]*self.dim
        self.ub = [0.]*self.dim

        for i in range(self.dim):
            self.lb[i] = self.hyperparams[i][0]
            self.ub[i] = self.hyperparams[i][1]

        # self.hypvol = np.subtract(self.ub,self.lb)

    # PDF
    def get_one_pdf_value(self,x,pos=0): 
        if x>=self.lb[pos] and x<=self.ub[pos]:
            return np.divide(1,self.ub[pos]-self.lb[pos])
        else:
            return 0.

    def get_pdf_values(self,x,pos=0):
        values = [0.]*len(x)
        for i in range(len(x)):
            values[i] += self.get_one_pdf_value(x[i],pos)
        return values

    # CDF
    def get_one_cdf_value(self,x,pos=0): 
        if x<self.lb[pos]:
            return 0.
        elif x>=self.lb[pos] and x<=self.ub[pos]:
            return np.divide(x-self.lb[pos],self.ub[pos]-self.lb[pos])
        else:
            return 1.

    def get_cdf_values(self,x,pos=0):
        values = [0.]*len(x)
        for i in range(len(x)):
            values[i] += self.get_one_cdf_value(x[i],pos)
        return values

    # Inverse CDF
    def inv_cdf(self,x,pos=0):
        if x<=0.:
            return self.lb[pos]
        elif x>=self.lb[pos] and x<=self.ub[pos]:
            return (self.ub[pos]-self.lb[pos])*x + self.lb[pos]
        else:
            return self.ub[pos]

    # Generate random samples
    def get_one_sample(self,pos=0):
        u = np.random.random()
        return self.inv_cdf(u,pos)

    def get_samples(self,nsamples,pos=0):
        values = [0.]*nsamples
        for i in range(nsamples):
            values[i] = self.get_one_sample(pos)

        return values