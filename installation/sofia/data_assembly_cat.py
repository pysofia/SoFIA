import numpy as np
import sofia.distributions as dist
import mutationpp as mpp
import pickle
import json
import sys

## String structure: case

class assembly:

    def __init__(self,data_json,models_json,mixture_object):
        with open(data_json) as jfile:
            self.cases = json.load(jfile)
        with open(models_json) as jfile_m:
            self.models = json.load(jfile_m)
        self.mix = mixture_object
        ##
        self.case = sys.argv[1]
        self.lik = pickle.load(open(self.models[self.case]["model"], 'rb'))

    ## Model-related assembly

    def assembly_hyperparameters(self):
        self.hyp = self.models[self.case]['priors']
        self.prior = dist.Uniform(len(self.hyp),self.hyp)

    def denormalization(self,Xi):
        Xi_denorm = [0.]*len(Xi)
        for i in range(len(self.hyp)):
            Xi_denorm[i] = self.prior.lb[i]+(Xi[i]*(self.prior.ub[i]-self.prior.lb[i]))
        return Xi_denorm