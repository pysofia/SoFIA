import numpy as np
from numpy import random

class data:

    def __init__(self,)

class likelihood:

    def __init__(self,dist,data,fun_in,params):
        self.data = data
        self.fun = fun_in
        self.par = params
        if dist == "Gaussian":
            self.dist = -np.power(np.abs(self.fun-self.fun(self.data,self.params)),2)/(2*np.power(sigma*ishigami(0,1,2,a,b),2))
        self.dist = 
