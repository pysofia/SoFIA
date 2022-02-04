import numpy as np
import sofia.distributions as dist
import mutationpp as mpp
import pickle
import json
import sys

class data():

    def __init__(self,json_file):
        self.cases = json.load(json_file)


    def ve_betae(self,X): # X = (T, pres, Pd)
    
        mix.equilibrate(X[2], X[1])
        rhoe = mix.density()

        Vs_ext = np.power(2*X[0]/1.1/rhoe,0.5) #; //Kp = 1.1
        
        ve = Vs_ext*self.cases[sys.argv[1]]["NDPs"][4]

        V_torch = ve/self.cases[sys.argv[1]]["NDPs"][3] #; //NDP_ve = NDP4
        betae = V_torch*self.cases[sys.argv[1]]["NDPs"][1]/0.025

        vect = np.zeros(2)
        vect[0] += ve
        vect[1] += betae

        return vect


class model(dist.Gaussian,dist.Uniform,data):

    def __init__(self,json_file,distr,likelihood):
        self.models = json.load(json_file)
        self.prior = distr
        self.hyp = self.prior.hyperparams
        self.Lik = likelihood

    def recession(self):
        rec = pickle.load(open(self.models[sys.argv[3]][sys.argv[4]][sys.argv[1]][0], 'rb'))
        return lambda u: rec.predict(u)[0]

    def density(self):
        rho = pickle.load(open(self.models[sys.argv[3]][sys.argv[4]][sys.argv[1]][1], 'rb'))
        return lambda u: rho.predict(u)[0]

    def wall_temp(self):
        temp = pickle.load(open(self.models[sys.argv[3]][sys.argv[4]][sys.argv[1]][2], 'rb'))
        return lambda u: temp.predict(u)[0]

    def hyp_priors(self):
        return self.models[sys.argv[3]][sys.argv[4]]['priors']

    def denormalization(self,Xi):
        Xi_denorm = [0.]*len(Xi)
        for i in range(len(self.hyp)):
            Xi_denorm[i] = self.prior.lb[i]+(Xi[i]*(self.prior.ub[i]-self.prior.lb[i]))
        return Xi_denorm

    def log_likelihood(self,Xi):
        for i in range(len(Xi)):
        if Xi[i]<0 or Xi[i]>1:
            return -1.e16

        X = np.zeros(3)
        X[0] += self.prior.lb[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Pd"]]+(Xi[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Pd"]]*(self.prior.ub[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Pd"]]-self.prior.lb[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Pd"]]))

        X[1] += self.prior.lb[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Ps"]]+(Xi[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Ps"]]*(self.prior.ub[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Ps"]]-self.prior.lb[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Ps"]]))

        X[2] += self.prior.lb[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Te"]]+(Xi[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Te"]]*(self.prior.ub[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Te"]]-self.prior.lb[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Te"]]))

        ve, betae = self.ve_betae(X)

        V = np.zeros(len(self.hyp)+1)
        V[0] += Xi[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Ps"]]
        if sys.argv[4]=="SEB":
            V[1] += Xi[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Te"]]
            V[2] += (ve - 300.)/900. 
            V[3] += (betae - 20000.)/44000.
            V[4] += Xi[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Gnit"]]
            V[5] += Xi[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Grec"]]
            if sys.argv[3]=="2T":
                V[6] += Xi[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["epsilon"]]
                V[7] += Xi[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["alpha"]]
                V[8] += Xi[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["beta"]]
            
        elif sys.argv[4]=="baseline":
            V[1] += Xi[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Tw"]]
            V[2] += Xi[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Te"]]
            V[3] += (ve - 300.)/900.
            V[4] += (betae - 20000.)/44000.
            V[5] += Xi[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Gnit"]]
        else:
            V[1] += Xi[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Tw"]]
            V[2] += Xi[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Te"]]
            V[3] += (ve - 300.)/900.
            V[4] += (betae - 20000.)/44000.
            V[5] += Xi[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Gnit"]]
            V[6] += Xi[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Grec"]]

        V = [V]

        Xi_denorm = self.denormalization(Xi)

        if sys.argv[4]=="SEB":

            if sys.argv[2] =="all":

                return self.Lik.get_one_prop_logpdf_value(Xi_denorm[models[sys.argv[3]][sys.argv[4]]["indexes"]["Ps"]],0) + self.Lik.get_one_prop_logpdf_value(Xi_denorm[models[sys.argv[3]][sys.argv[4]]["indexes"]["Pd"]],2) + self.Lik.get_one_prop_logpdf_value(np.power(10,self.recession(V)),3)+self.Lik.get_one_prop_logpdf_value(np.power(10,self.density(V)),4)+self.Lik.get_one_prop_logpdf_value(self.wall_temp(V),5)

            elif sys.argv[2] =="recession":

                return self.Lik.get_one_prop_logpdf_value(Xi_denorm[models[sys.argv[3]][sys.argv[4]]["indexes"]["Ps"]],0) + self.Lik.get_one_prop_logpdf_value(Xi_denorm[models[sys.argv[3]][sys.argv[4]]["indexes"]["Pd"]],2) + self.Lik.get_one_prop_logpdf_value(np.power(10,self.recession(V)),3)+self.Lik.get_one_prop_logpdf_value(self.wall_temp(V),5)

            else:

                return self.Lik.get_one_prop_logpdf_value(Xi_denorm[models[sys.argv[3]][sys.argv[4]]["indexes"]["Ps"]],0) + self.Lik.get_one_prop_logpdf_value(Xi_denorm[models[sys.argv[3]][sys.argv[4]]["indexes"]["Pd"]],2) + self.Lik.get_one_prop_logpdf_value(np.power(10,self.density(V)),4)+self.Lik.get_one_prop_logpdf_value(self.wall_temp(V),5)

        else:

            if sys.argv[2] =="all":

                return self.Lik.get_one_prop_logpdf_value(Xi_denorm[models[sys.argv[3]][sys.argv[4]]["indexes"]["Ps"]],0) + self.Lik.get_one_prop_logpdf_value(Xi_denorm[models[sys.argv[3]][sys.argv[4]]["indexes"]["Tw"]],1) + self.Lik.get_one_prop_logpdf_value(Xi_denorm[models[sys.argv[3]][sys.argv[4]]["indexes"]["Pd"]],2) + self.Lik.get_one_prop_logpdf_value(np.power(10,self.recession(V)),3)+self.Lik.get_one_prop_logpdf_value(np.power(10,self.density(V)),4)

            elif sys.argv[2] =="recession":

                return self.Lik.get_one_prop_logpdf_value(Xi_denorm[models[sys.argv[3]][sys.argv[4]]["indexes"]["Ps"]],0) + self.Lik.get_one_prop_logpdf_value(Xi_denorm[models[sys.argv[3]][sys.argv[4]]["indexes"]["Tw"]],1) + self.Lik.get_one_prop_logpdf_value(Xi_denorm[models[sys.argv[3]][sys.argv[4]]["indexes"]["Pd"]],2) + self.Lik.get_one_prop_logpdf_value(np.power(10,self.recession(V)),3)

            else:

                return self.Lik.get_one_prop_logpdf_value(Xi_denorm[models[sys.argv[3]][sys.argv[4]]["indexes"]["Ps"]],0) + self.Lik.get_one_prop_logpdf_value(Xi_denorm[models[sys.argv[3]][sys.argv[4]]["indexes"]["Tw"]],1) + self.Lik.get_one_prop_logpdf_value(Xi_denorm[models[sys.argv[3]][sys.argv[4]]["indexes"]["Pd"]],2) + self.Lik.get_one_prop_logpdf_value(np.power(10,self.density(V)),4)

    def m_log_likelihood(self,Xi):
        return -1*self.log_likelihood(Xi)