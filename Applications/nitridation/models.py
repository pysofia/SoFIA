import numpy as np
import sofia.distributions as dist
import mutationpp as mpp
import pickle
import json
import sys

class data(dist.Gaussian):

    """

        This class loads and creates the data-related objects for the Bayesian inference of nitridation reaction efficiencies.

        INPUTS: json file with the data of each case

        OUTPUTS: defines the parameters associated with the experimental data in the likelihood: mean values and std devs.

    """

    def __init__(self,json_file):
        with open(json_file) as jfile:
            self.cases = json.load(jfile)

    # def define_likelihood(self):

    #     if sys.argv[2] =='all':
    #         h = [[1500.,22.5],[self.cases[sys.argv[1]]['Tw']['mean'],self.cases[sys.argv[1]]['Tw']['std-dev']],[self.cases[sys.argv[1]]['Pd']['mean'],self.cases[sys.argv[1]]['Pd']['std-dev']],[self.cases[sys.argv[1]]['recession']['mean'],self.cases[sys.argv[1]]['recession']['std-dev']],[self.cases[sys.argv[1]]['density']['mean'],self.cases[sys.argv[1]]['density']['std-dev']]]
    #     else:
    #         h = [[1500.,22.5],[self.cases[sys.argv[1]]['Tw']['mean'],self.cases[sys.argv[1]]['Tw']['std-dev']],[self.cases[sys.argv[1]]['Pd']['mean'],self.cases[sys.argv[1]]['Pd']['std-dev']],[self.cases[sys.argv[1]][sys.argv[2]]['mean'],self.cases[sys.argv[1]][sys.argv[2]]['std-dev']]]

    #     self.Lik = dist.Gaussian(len(h),h)

    def define_likelihood(self):

        if sys.argv[1] == "Arrhenius":

            cases = ["G4","G5","G6","G7"]

            if sys.argv[2] =='all':
                h = []
                for case in cases:
                    h.append([1500.,22.5])
                    h.append([self.cases[case]['Tw']['mean'],self.cases[case]['Tw']['std-dev']])
                    h.append([self.cases[case]['Pd']['mean'],self.cases[case]['Pd']['std-dev']])
                    h.append([self.cases[case]['recession']['mean'],self.cases[case]['recession']['std-dev']])
                    h.append([self.cases[case]['density']['mean'],self.cases[case]['density']['std-dev']])
            else:
                h = []
                for case in cases:
                    h.append([1500.,22.5])
                    h.append([self.cases[case]['Tw']['mean'],self.cases[case]['Tw']['std-dev']])
                    h.append([self.cases[case]['Pd']['mean'],self.cases[case]['Pd']['std-dev']])
                    h.append([self.cases[case][sys.argv[2]]['mean'],self.cases[case][sys.argv[2]]['std-dev']])

        else:

            if sys.argv[2] =='all':
                h = [[1500.,22.5],[self.cases[sys.argv[1]]['Tw']['mean'],self.cases[sys.argv[1]]['Tw']['std-dev']],[self.cases[sys.argv[1]]['Pd']['mean'],self.cases[sys.argv[1]]['Pd']['std-dev']],[self.cases[sys.argv[1]]['recession']['mean'],self.cases[sys.argv[1]]['recession']['std-dev']],[self.cases[sys.argv[1]]['density']['mean'],self.cases[sys.argv[1]]['density']['std-dev']]]
            else:
                h = [[1500.,22.5],[self.cases[sys.argv[1]]['Tw']['mean'],self.cases[sys.argv[1]]['Tw']['std-dev']],[self.cases[sys.argv[1]]['Pd']['mean'],self.cases[sys.argv[1]]['Pd']['std-dev']],[self.cases[sys.argv[1]][sys.argv[2]]['mean'],self.cases[sys.argv[1]][sys.argv[2]]['std-dev']]]

        self.Lik = dist.Gaussian(len(h),h)


class model:

    """

        This class loads and creates the model-related objects for the Bayesian inference of nitridation reaction efficiencies.

        INPUTS: json files with the data of each case and model files, mpp mixture and the defined likelihood with the experimental data.

        OUTPUTS: recession rate, CN density and wall temperature models, prior distributions of the corresponding parameters, denormalization of the parameters if working with canonical variables, ve and betae function definition with the NDPs stored in the data json file, log likelihood and minus log likelihood functions.

    """

    def __init__(self,data,json_file_models,mixture,likelihood):
        with open(json_file_models) as jfile_m:
            self.models = json.load(jfile_m)
        self.cases = data
        self.Lik = likelihood
        self.mix = mixture
        ##
        self.rec = pickle.load(open(self.models[sys.argv[3]][sys.argv[4]]["G4"][0], 'rb'))
        self.temp = pickle.load(open(self.models[sys.argv[3]]["SEB"]["G4"][2], 'rb'))

    # def recession(self):
    #     rec = pickle.load(open(self.models[sys.argv[3]][sys.argv[4]]["G4"][0], 'rb'))
    #     return lambda u: rec.predict(u)[0]

    def recession(self):
        return lambda u: self.rec.predict(u)[0]

    def density(self,case=None):
        if case != None:
            rho = pickle.load(open(self.models[sys.argv[3]][sys.argv[4]][case][1], 'rb'))
        else:
            rho = pickle.load(open(self.models[sys.argv[3]][sys.argv[4]][sys.argv[1]][1], 'rb'))
        return lambda u: rho.predict(u)[0]

    # def wall_temp(self):
    #     temp = pickle.load(open(self.models[sys.argv[3]][sys.argv[4]]["G4"][2], 'rb'))
    #     return lambda u: temp.predict(u)[0]

    def wall_temp(self):
        return lambda u: self.temp.predict(u)[0]

    # def define_prior(self):
    #     self.hyp = self.models[sys.argv[3]][sys.argv[4]]['priors']
    #     self.prior = dist.Uniform(len(self.hyp),self.hyp)

    def define_prior(self):
        if sys.argv[1] == "Arrhenius":

            cases = ["G4","G5","G6","G7"]
            self.hyp = []

            for case in cases:
                for i in range(1,len(self.models[sys.argv[3]][sys.argv[4]]['priors'])):
                    self.hyp.append(self.models[sys.argv[3]][sys.argv[4]]['priors'][i])
            self.hyp.append(self.models['Arrhenius_priors'][self.models['Arrhenius_indexes']["A"]])
            self.hyp.append(self.models['Arrhenius_priors'][self.models['Arrhenius_indexes']["Ta"]])
            self.prior = dist.Uniform(len(self.hyp),self.hyp)
        else: 
            self.hyp = self.models[sys.argv[3]][sys.argv[4]]['priors']
            self.prior = dist.Uniform(len(self.hyp),self.hyp)

    def denormalization(self,Xi):
        Xi_denorm = [0.]*len(Xi)
        for i in range(len(self.hyp)):
            Xi_denorm[i] = self.prior.lb[i]+(Xi[i]*(self.prior.ub[i]-self.prior.lb[i]))
        return Xi_denorm

    def ve_betae(self,X,case=None): # X = (T, pres, Pd)

        if case != None:
            cas = case
        else:
            cas = sys.argv[1]
    
        self.mix.equilibrate(X[2], X[1])
        rhoe = self.mix.density()

        Vs_ext = np.power(2*X[0]/1.1/rhoe,0.5) #; //Kp = 1.1
        
        ve = Vs_ext*self.cases[cas]["NDPs"][4]

        V_torch = ve/self.cases[cas]["NDPs"][3] #; //NDP_ve = NDP4
        betae = V_torch*self.cases[cas]["NDPs"][1]/0.025

        vect = np.zeros(2)
        vect[0] += ve
        vect[1] += betae

        return vect

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

                return self.Lik.get_one_prop_logpdf_value(Xi_denorm[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Ps"]],0) + self.Lik.get_one_prop_logpdf_value(Xi_denorm[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Pd"]],2) + self.Lik.get_one_prop_logpdf_value(np.power(10,self.recession()(V)),3)+self.Lik.get_one_prop_logpdf_value(np.power(10,self.density()(V)),4)+self.Lik.get_one_prop_logpdf_value(self.wall_temp()(V),1)

            elif sys.argv[2] =="recession":

                return self.Lik.get_one_prop_logpdf_value(Xi_denorm[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Ps"]],0) + self.Lik.get_one_prop_logpdf_value(Xi_denorm[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Pd"]],2) + self.Lik.get_one_prop_logpdf_value(np.power(10,self.recession()(V)),3)+self.Lik.get_one_prop_logpdf_value(self.wall_temp()(V),1)

            else:

                return self.Lik.get_one_prop_logpdf_value(Xi_denorm[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Ps"]],0) + self.Lik.get_one_prop_logpdf_value(Xi_denorm[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Pd"]],2) + self.Lik.get_one_prop_logpdf_value(np.power(10,self.density()(V)),3)+self.Lik.get_one_prop_logpdf_value(self.wall_temp()(V),1)

        else:

            if sys.argv[2] =="all":

                return self.Lik.get_one_prop_logpdf_value(Xi_denorm[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Ps"]],0) + self.Lik.get_one_prop_logpdf_value(Xi_denorm[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Tw"]],1) + self.Lik.get_one_prop_logpdf_value(Xi_denorm[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Pd"]],2) + self.Lik.get_one_prop_logpdf_value(np.power(10,self.recession()(V)),3)+self.Lik.get_one_prop_logpdf_value(np.power(10,self.density()(V)),4)

            elif sys.argv[2] =="recession":

                return self.Lik.get_one_prop_logpdf_value(Xi_denorm[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Ps"]],0) + self.Lik.get_one_prop_logpdf_value(Xi_denorm[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Tw"]],1) + self.Lik.get_one_prop_logpdf_value(Xi_denorm[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Pd"]],2) + self.Lik.get_one_prop_logpdf_value(np.power(10,self.recession()(V)),3)

            else:

                return self.Lik.get_one_prop_logpdf_value(Xi_denorm[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Ps"]],0) + self.Lik.get_one_prop_logpdf_value(Xi_denorm[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Tw"]],1) + self.Lik.get_one_prop_logpdf_value(Xi_denorm[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Pd"]],2) + self.Lik.get_one_prop_logpdf_value(np.power(10,self.density()(V)),3)

    def m_log_likelihood(self,Xi):
        return -1*self.log_likelihood(Xi)

    def log_likelihood_Arrhenius(self,Xi):
        for i in range(len(Xi)):
            if Xi[i]<0 or Xi[i]>1:
                return -1.e16

        cases = ["G4","G5","G6","G7"]
        value = 0.
        n_variables_in_each_case = len(self.models[sys.argv[3]][sys.argv[4]]["indexes"])

        for i in range(len(cases)):

            X = np.zeros(3)
            X[0] += self.prior.lb[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Pd"]-1+n_variables_in_each_case*i]+(Xi[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Pd"]-1+n_variables_in_each_case*i]*(self.prior.ub[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Pd"]-1+n_variables_in_each_case*i]-self.prior.lb[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Pd"]-1+n_variables_in_each_case*i]))

            X[1] += self.prior.lb[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Ps"]-1+n_variables_in_each_case*i]+(Xi[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Ps"]-1+n_variables_in_each_case*i]*(self.prior.ub[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Ps"]-1+n_variables_in_each_case*i]-self.prior.lb[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Ps"]-1+n_variables_in_each_case*i]))

            X[2] += self.prior.lb[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Te"]-1+n_variables_in_each_case*i]+(Xi[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Te"]-1+n_variables_in_each_case*i]*(self.prior.ub[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Te"]-1+n_variables_in_each_case*i]-self.prior.lb[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Te"]-1+n_variables_in_each_case*i]))

            ve, betae = self.ve_betae(X,cases[i])

            V = np.zeros(n_variables_in_each_case+1)
            V[0] += Xi[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Ps"]+n_variables_in_each_case*i]
            if sys.argv[4]=="SEB":
                V[1] += Xi[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Te"]+n_variables_in_each_case*i]
                V[2] += (ve - 300.)/900. 
                V[3] += (betae - 20000.)/44000.

                if sys.argv[1] == "Arrhenius":
                    V[4] += Xi[-2]*np.exp(-1*np.divide(Xi[-1],self.wall_temp()(V)))
                else:
                    V[4] += Xi[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Gnit"]+n_variables_in_each_case*i]

                V[5] += Xi[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Grec"]+n_variables_in_each_case*i]
                if sys.argv[3]=="2T":
                    V[6] += Xi[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["epsilon"]+n_variables_in_each_case*i]
                    V[7] += Xi[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["alpha"]+n_variables_in_each_case*i]
                    V[8] += Xi[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["beta"]+n_variables_in_each_case*i]
                
            elif sys.argv[4]=="baseline":
                V[1] += Xi[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Tw"]+n_variables_in_each_case*i]
                V[2] += Xi[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Te"]+n_variables_in_each_case*i]
                V[3] += (ve - 300.)/900.
                V[4] += (betae - 20000.)/44000.
                if sys.argv[1] == "Arrhenius":
                    V[5] += Xi[-2]*np.exp(-1*np.divide(Xi[-1],self.wall_temp()(V)))
                else:
                    V[5] += Xi[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Gnit"]+n_variables_in_each_case*i]
            else:
                V[1] += Xi[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Tw"]+n_variables_in_each_case*i]
                V[2] += Xi[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Te"]+n_variables_in_each_case*i]
                V[3] += (ve - 300.)/900.
                V[4] += (betae - 20000.)/44000.
                if sys.argv[1] == "Arrhenius":
                    V[5] += np.power(10.,-4.)*np.exp(-1*np.divide(5000.,2000.)) #Xi[-2]*np.exp(-1*np.divide(Xi[-1],2000.))
                else:
                    V[5] += Xi[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Gnit"]+n_variables_in_each_case*i]
                V[6] += Xi[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Grec"]+n_variables_in_each_case*i]

            V = [V]

            Xi_denorm = self.denormalization(Xi)

            if sys.argv[4]=="SEB":

                if sys.argv[2] =="all":

                    value += self.Lik.get_one_prop_logpdf_value(Xi_denorm[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Ps"]+n_variables_in_each_case*i],0+n_variables_in_each_case*i) + self.Lik.get_one_prop_logpdf_value(Xi_denorm[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Pd"]+n_variables_in_each_case*i],2+n_variables_in_each_case*i) + self.Lik.get_one_prop_logpdf_value(np.power(10,self.recession()(V)),3+n_variables_in_each_case*i)+self.Lik.get_one_prop_logpdf_value(np.power(10,self.density(cases[i])(V)),4+n_variables_in_each_case*i)+self.Lik.get_one_prop_logpdf_value(self.wall_temp()(V),1+n_variables_in_each_case*i)

                elif sys.argv[2] =="recession":

                    value += self.Lik.get_one_prop_logpdf_value(Xi_denorm[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Ps"]+n_variables_in_each_case*i],0+n_variables_in_each_case*i) + self.Lik.get_one_prop_logpdf_value(Xi_denorm[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Pd"]+n_variables_in_each_case*i],2+n_variables_in_each_case*i) + self.Lik.get_one_prop_logpdf_value(np.power(10,self.recession()(V)),3+n_variables_in_each_case*i)+self.Lik.get_one_prop_logpdf_value(self.wall_temp()(V),1+n_variables_in_each_case*i)

                else:

                    value += self.Lik.get_one_prop_logpdf_value(Xi_denorm[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Ps"]+n_variables_in_each_case*i],0+n_variables_in_each_case*i) + self.Lik.get_one_prop_logpdf_value(Xi_denorm[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Pd"]+n_variables_in_each_case*i],2+n_variables_in_each_case*i) + self.Lik.get_one_prop_logpdf_value(np.power(10,self.density(cases[i])(V)),3+n_variables_in_each_case*i)+self.Lik.get_one_prop_logpdf_value(self.wall_temp()(V),1+n_variables_in_each_case*i)

            else:

                if sys.argv[2] =="all":

                    value += self.Lik.get_one_prop_logpdf_value(Xi_denorm[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Ps"]+n_variables_in_each_case*i],0+n_variables_in_each_case*i) + self.Lik.get_one_prop_logpdf_value(Xi_denorm[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Tw"]+n_variables_in_each_case*i],1+n_variables_in_each_case*i) + self.Lik.get_one_prop_logpdf_value(Xi_denorm[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Pd"]+n_variables_in_each_case*i],2+n_variables_in_each_case*i) + self.Lik.get_one_prop_logpdf_value(np.power(10,self.recession()(V)),3+n_variables_in_each_case*i)+self.Lik.get_one_prop_logpdf_value(np.power(10,self.density(cases[i])(V)),4+n_variables_in_each_case*i)

                elif sys.argv[2] =="recession":

                    value += self.Lik.get_one_prop_logpdf_value(Xi_denorm[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Ps"]-1+n_variables_in_each_case*i],0+n_variables_in_each_case*i) + self.Lik.get_one_prop_logpdf_value(Xi_denorm[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Tw"]-1+n_variables_in_each_case*i],1+n_variables_in_each_case*i) + self.Lik.get_one_prop_logpdf_value(Xi_denorm[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Pd"]-1+n_variables_in_each_case*i],2+n_variables_in_each_case*i) + self.Lik.get_one_prop_logpdf_value(np.power(10,self.recession()(V)),3+n_variables_in_each_case*i)

                else:

                    value += self.Lik.get_one_prop_logpdf_value(Xi_denorm[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Ps"]+n_variables_in_each_case*i],0+n_variables_in_each_case*i) + self.Lik.get_one_prop_logpdf_value(Xi_denorm[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Tw"]+n_variables_in_each_case*i],1+n_variables_in_each_case*i) + self.Lik.get_one_prop_logpdf_value(Xi_denorm[self.models[sys.argv[3]][sys.argv[4]]["indexes"]["Pd"]+n_variables_in_each_case*i],2+n_variables_in_each_case*i) + self.Lik.get_one_prop_logpdf_value(np.power(10,self.density(cases[i])(V)),3+n_variables_in_each_case*i)

        return value

    def m_log_likelihood_Arrhenius(self,Xi):
        return -1*self.log_likelihood_Arrhenius(Xi)