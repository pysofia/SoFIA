import sys
import scipy
from scipy.optimize import minimize
import numpy as np
import subprocess
from subprocess import call
from pyDOE import *
from operator import itemgetter

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

def functional(p0,m,u,ref,gamma,gammaref,gamma_qz):

    x0=p0*ref #Des-normalization
   
    x1=[x0[0],x0[1],x0[2],x0[3]]
    x2=[x0[0],x0[1],x0[4],x0[3]]
    x3=[x0[0],x0[1],x0[5],x0[3]]
    
    L = np.divide(np.absolute(m[0] - BL(x1,gamma)) ** 2, 2 * u[0] ** 2)+np.divide(np.absolute(m[5] - BL(x2,gammaref)) 	 ** 2, 2 * u[5] ** 2)+np.divide(np.absolute(m[7]-BL(x3,gamma_qz)) ** 2, 2 * u[7] ** 2)+np.divide(np.absolute(m[6]-x0[5]) ** 2, 2 * u[6] ** 2)

    for i in range (1,5):

        L=L+np.divide(np.absolute(m[i]-x0[i])**2,2*u[i]**2)


    print(x0,L)
    print('gamma=', gamma, 'gammaref=', gammaref, 'gamma_qz=', gamma_qz)

    return L

# ---------------------------------
#Subfunctional to minimize

def subfunctional(p0S,m,u,refS,gamma,gammaref,gamma_qz):

    x0S=p0S*refS #Des-normalization
   
    x1=[x0S[0],m[1],m[2],m[3]]
    x2=[x0S[0],m[1],m[4],m[3]]
    x3=[x0S[0],m[1],m[6],m[3]]

    SL = np.divide(np.absolute(m[0] - BL(x1,gamma)) ** 2, 2 * u[0] ** 2)+np.divide(np.absolute(m[5] - BL(x2,gammaref)) 	 ** 2, 2 * u[5] ** 2)+np.divide(np.absolute(m[7]-BL(x3,gamma_qz)) ** 2, 2 * u[7] ** 2)


    print(x0S,SL)
    print('gamma=', gamma, 'gammaref=', gammaref, 'gamma_qz=', gamma_qz)
    return SL

# ---------------------------------
#Minimization algorithm


#LHS sampling----------------------

chain=np.loadtxt('./chain_gp2.gnu')

xp=np.log10(chain[:,0])
yp=np.log10(chain[:,1])
zp=np.log10(chain[:,2])

#gamma1=lhs(3,samples=40)
#gamma1=np.random.random_sample((10,3))

#gamma = sorted(-gamma1, key=itemgetter(0))

#gamma_qz1=[0.]*len(gamma1)
#gamma_cu1=[0.]*len(gamma1)
#gamma_ag1=[0.]*len(gamma1)

gamma_qz1=[0.]*20
gamma_cu1=[0.]*20
gamma_ag1=[0.]*20

for i in range(20):
 #gamma_qz1[i]=(max(xp)-min(xp))*-gamma[i][0]+min(xp)
 #gamma_cu1[i]=(max(zp)-min(zp))*-gamma[i][1]+min(zp)
 #gamma_ag1[i]=(max(yp)-min(yp))*-gamma[i][2]+min(yp)
 gamma_qz1[i]=(max(xp)-min(xp))*np.random.random_sample()+min(xp)
 gamma_cu1[i]=0.0
 gamma_ag1[i]=(max(yp)-min(yp))*np.random.random_sample()+min(yp)

#Measurements, uncertainties and i. cond----------------------

#9B
#m = [2003590.0, 5000.0, 350.0, 177.18, 350.0, 2369670.0, 750.0, 795140.0]  # Put these two in file
#u = [89960.0, 30.0, 17.5, 1.97, 17.5, 106280.0, 37.5, 35860.0]

#ref = [37951903.47, 5017.5, 305.1416, 176.7, 298.8, 835.78]  # Normalization

#3A
#m = [700000.0, 1500.0, 350.0, 181.74, 350.0, 720694.0, 750.0, 273236.0]  # Put these two in file
#u = [31080.0, 9.0, 17.5, 2.76, 17.5, 31998.81, 37.5, 12377.6]

#ref = [18000000.0, 1700.0, 340.0, 170.74, 360.0, 800.0]  # Normalization

m = [2000000.0, 10000.0, 350.0, 60.9, 350.0, 2203810.0, 750.0, 897651.0]  # Put these two in file
u = [88800.0, 60.0, 17.5, 0.92, 17.5, 97849.16, 37.5, 40663.6]

ref = [23000000.0, 11000.0, 340.0, 70.0, 360.0, 800.0]  # Normalization

p0S = 0.8

refS = 37951903.47  # Normalization

p0 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

#Main loop----------------------

with open('./output.out', 'w') as out:

 for gammas in zip(gamma_qz1,gamma_cu1,gamma_ag1):
     
     gamma_true = np.power(10, gammas)
     
     #gamma_true=[0.00400464907795,0.0139797170591,0.59561893937]

     #res0=scipy.optimize.minimize(subfunctional,p0S,args=(m,u,refS,gamma_true[1],gamma_true[2],gamma_true[0]),method='Nelder-Mead')

     #p0 = [res0.x, 1.0, 1.0, 1.0, 1.0, 1.0]

     res=scipy.optimize.minimize(functional,p0,args=(m,u,ref,gamma_true[1],gamma_true[2],gamma_true[0]),method='Nelder-Mead') #options={'xatol': 0.0001, 'fatol': 0.0001})

     x=[res.x[0]*ref[0],res.x[1]*ref[1],res.x[2]*ref[2],res.x[3]*ref[3],res.x[4]*ref[4],res.x[5]*ref[5]]

     out.write(' '.join(str(out) for out in x+[np.exp(-res.fun)]+[res.fun]+[np.log10(gamma_true[0])]+[np.log10(gamma_true[2])]+[np.log10(gamma_true[1])])+'\n')
     
