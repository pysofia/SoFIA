import numpy as np
import sofia.distributions as dist
import mutationpp as mpp
import pickle
import json
import sys

## String structure: case-thermo_model-recombination/baseline/SEB

class assembly:

    def __init__(self,data_json,models_json,mixture_object):
        with open(data_json) as jfile:
            self.cases = json.load(jfile)
        with open(models_json) as jfile_m:
            self.models = json.load(jfile_m)
        self.mix = mixture_object
        ##
        self.case = sys.argv[1]
        self.therm_model = sys.argv[2]
        self.model = sys.argv[3]
        self.rec = pickle.load(open(self.models[self.therm_model][self.model][self.case][0], 'rb'))
        self.temp = pickle.load(open(self.models[self.therm_model]["SEB"][self.case][2], 'rb'))
        self.rho = pickle.load(open(self.models[self.therm_model][self.model][self.case][1], 'rb'))

    ## Experimental data-related assembly

    def lik_hyperparams(self,measurements):
        h = [[self.cases[self.case][measurements[i]]['mean'],self.cases[self.case][measurements[i]]['std-dev']] for i in range(len(measurements))]

        self.Lik = dist.Gaussian(len(h),h)
        self.measurements = measurements

    ## Model-related assembly

    def recession(self):
        return lambda u: self.rec.predict(u)[0]

    def wall_temp(self):
        return lambda u: self.temp.predict(u)[0]

    def density(self):
        return lambda u: self.rho.predict(u)[0]

    def assembly_prior(self):
        self.hyp = self.models[self.therm_model][self.model]['priors']
        self.prior = dist.Uniform(len(self.hyp),self.hyp)

    def denormalization(self,Xi):
        Xi_denorm = [0.]*len(Xi)
        for i in range(len(self.hyp)):
            Xi_denorm[i] = self.prior.lb[i]+(Xi[i]*(self.prior.ub[i]-self.prior.lb[i]))
        return Xi_denorm

    def ve_betae(self,X): # X = (T, pres, Pd)
    
        self.mix.equilibrate(X[2], X[1])
        rhoe = self.mix.density()

        Vs_ext = np.power(2*X[0]/1.1/rhoe,0.5) #; //Kp = 1.1
        
        ve = Vs_ext*self.cases[self.case]["NDPs"][4]

        V_torch = ve/self.cases[self.case]["NDPs"][3] #; //NDP_ve = NDP4
        betae = V_torch*self.cases[self.case]["NDPs"][1]/0.025

        vect = np.zeros(2)
        vect[0] += ve
        vect[1] += betae

        return vect
    
    def log_likelihood(self,Xi):
        for i in range(len(Xi)):
            if Xi[i]<0 or Xi[i]>1:
                return -1.e16

        X = np.zeros(3)
        X[0] += self.prior.lb[self.models[self.therm_model][self.model]["indexes"]["Pd"]]+(Xi[self.models[self.therm_model][self.model]["indexes"]["Pd"]]*(self.prior.ub[self.models[self.therm_model][self.model]["indexes"]["Pd"]]-self.prior.lb[self.models[self.therm_model][self.model]["indexes"]["Pd"]]))

        X[1] += self.prior.lb[self.models[self.therm_model][self.model]["indexes"]["Ps"]]+(Xi[self.models[self.therm_model][self.model]["indexes"]["Ps"]]*(self.prior.ub[self.models[self.therm_model][self.model]["indexes"]["Ps"]]-self.prior.lb[self.models[self.therm_model][self.model]["indexes"]["Ps"]]))

        X[2] += self.prior.lb[self.models[self.therm_model][self.model]["indexes"]["Te"]]+(Xi[self.models[self.therm_model][self.model]["indexes"]["Te"]]*(self.prior.ub[self.models[self.therm_model][self.model]["indexes"]["Te"]]-self.prior.lb[self.models[self.therm_model][self.model]["indexes"]["Te"]]))

        ve, betae = self.ve_betae(X)

        V = np.zeros(len(self.hyp)+1)
        for i in self.models[self.therm_model][self.model]["inputs"]:
            if i=="ve":
                V[self.models[self.therm_model][self.model]["inputs"][i]] = (ve - 300.)/900.
            elif i=="betae":
                V[self.models[self.therm_model][self.model]["inputs"][i]] = (betae - 20000.)/44000.
            else:
                V[self.models[self.therm_model][self.model]["inputs"][i]] += Xi[self.models[self.therm_model][self.model]["indexes"][i]]

        V = [V]

        Xi_denorm = self.denormalization(Xi)

        self.log_lik = 0.
        for i in range(len(self.measurements)):
            if self.measurements[i]=="recession":
                self.log_lik += self.Lik.get_one_prop_logpdf_value(np.power(10,self.recession()(V)),i)
            elif self.measurements[i]=="density":
                self.log_lik += self.Lik.get_one_prop_logpdf_value(np.power(10,self.density()(V)),i)
            elif self.measurements[i]=="Tw" and self.model=="SEB":
                self.log_lik += self.Lik.get_one_prop_logpdf_value(self.wall_temp()(V),i)
            else:
                self.log_lik += self.Lik.get_one_prop_logpdf_value(Xi_denorm[self.models[self.therm_model][self.model]["indexes"][self.measurements[i]]],i)

        return self.log_lik

    def m_log_likelihood(self,Xi):
        return -1*self.log_likelihood(Xi)