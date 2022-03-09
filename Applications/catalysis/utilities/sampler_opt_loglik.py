import sys
import scipy
from scipy.optimize import minimize
import numpy as np
import subprocess
from subprocess import call
from pyDOE import *
from operator import itemgetter
import json
import sofia.distributions as dist

# ---------------------------------
#BL Code function

def BL(x0,gamma):

    #---------------------------------
    # Write gamma for BL code on wall conditions

    with open('./neboulainput/bc_wall.in', 'r') as f:
        lines = f.readlines()

    # Replace the lines
    lines[3] = ('{:f}  \n').format(gamma)
    lines[5] = ('{:f}  \n').format(gamma)

    # Write the lines back out
    with open('./neboulainput/bc_wall.in', 'w') as f:
        f.writelines(lines)

    #---------------------------------
    #Write input for BL code on out conditions

    with open('./cerbereinput/input.in','r') as f:
        lines = f.readlines()

    # Replace the lines
    lines[1] = ('{:f} \n').format(x0[0]) #Hs
    lines[2] = ('{:f} \n').format(x0[1]) #Ps
    lines[3] = ('{:f} \n').format(x0[3]) #Pd
    lines[4] = ('{:f}  {:f} \n').format(0.025,x0[2]) #Reff,Tw

    # Write the lines back out
    with open('./cerbereinput/input.in', 'w') as f:
	    f.writelines(lines)

    # ---------------------------------
    # Execute code to write Neboula input

    call(["../../cerboula/exe/cerbere.exe"])

    # ---------------------------------
    # Execute BL code

    call(["../../cerboula/exe/neboula.exe"])

    # ---------------------------------
    # Read output from BL code

    with open('./neboulaoutput/heatskin.dat') as f:
        lines=f.readlines()
        for line in lines:
            qw_array = np.fromstring(line, dtype=float, sep=' ')

    qw=qw_array[7]

    print(qw)

    return qw

# ---------------------------------
#Functional to minimize

def mlog_lik(p0,ref,gamma,obj_qw,obj):
    x0=p0*ref #De-normalization
   
    x1=[x0[0],x0[1],x0[2],x0[3]]
    x2=[x0[0],x0[1],x0[4],x0[3]]
    x3=[x0[0],x0[1],x0[5],x0[3]]

    qw = [BL(x1,gamma[0]),BL(x2,gamma[1]),BL(x3,gamma[2])] 
    mlk = 0.
    for i in range(len(qw)):
        mlk += obj_qw.get_one_prop_logpdf_value(qw[i],pos=i)

    for i in range(1:len(x0)):
        mlk += obj.get_one_prop_logpdf_value(x0[i],pos=i) # Where's H x0[0]

    return -1*mlk

# ---------------------------------
#Assembly of likelihood

data_file = "../cases.json"
case = sys.argv[1]

with open(data_file) as jfile_m:
    data = json.load(jfile_m)

h_qw = [[data[case]['qw_cu']['mean'],data[case]['qw_cu']['std-dev']],[data[case]['qw_qz']['mean'],data[case]['qw_qz']['std-dev']],[data[case]['qw_TPS']['mean'],data[case]['qw_TPS']['std-dev']]]

Lik_qw = dist.Gaussian(len(h_qw),h_qw)

h = [[data[case]['Tw_cu']['mean'],data[case]['Tw_cu']['std-dev']],[data[case]['Tw_qz']['mean'],data[case]['Tw_qz']['std-dev']],[data[case]['Tw_TPS']['mean'],data[case]['Tw_TPS']['std-dev']],[data[case]['Pd']['mean'],data[case]['Pd']['std-dev']],[data[case]['Ps']['mean'],data[case]['Ps']['std-dev']]]

Lik = dist.Gaussian(len(h),h)


#LHS sampling and normalization----------------------

gamma=lhs(3,samples=100)

ref = [23000000.0, 11000.0, 340.0, 70.0, 360.0, 800.0]  # Normalization

p0 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

#Main loop----------------------

with open('./output.out', 'w') as out:

 for i in range(len(gamma)):
     
     gamma_true = np.power(10, gamma[i,:])

     res=scipy.optimize.minimize(mlog_lik,p0,args=(ref,gamma_true,Lik_qw,Lik),method='Nelder-Mead') #options={'xatol': 0.0001, 'fatol': 0.0001})

     x=[res.x[0]*ref[0],res.x[1]*ref[1],res.x[2]*ref[2],res.x[3]*ref[3],res.x[4]*ref[4],res.x[5]*ref[5]]

     out.write(' '.join(str(out) for out in x+[np.exp(-res.fun)]+[res.fun]+[np.log10(gamma_true[0])]+[np.log10(gamma_true[2])]+[np.log10(gamma_true[1])])+'\n')
     
