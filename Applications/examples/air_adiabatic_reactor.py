import mutationpp as mpp
import sofia.pc as pce
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
import sofia.distributions as dist
import sofia.Sobol as sbl
import os, shutil
import sys

# Time integration

def runge_kutta4(mix, rhoi, T, dt):
    
    mix.setState(rhoi,T,0)
    wdot_1 = mix.netProductionRates() 

    rhoi += 0.5 * np.multiply(dt, wdot_1)
    mix.setState(rhoi,T,0)
    wdot_2 = mix.netProductionRates() 
    
    rhoi += 0.5 * np.multiply(dt, wdot_2)
    mix.setState(rhoi,T,0)
    wdot_3 = mix.netProductionRates() 
    
    rhoi += np.multiply(dt,wdot_3)
    mix.setState(rhoi,T,0)
    wdot_4 = mix.netProductionRates() 

    return 1./6. * np.multiply(dt, (np.array(wdot_1) + 2 * np.array(wdot_2) + 2 * np.array(wdot_3) + np.array(wdot_4)))

# Plotting settings

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 18

plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
cc = cycler('linestyle', ['-', '--', ':', '-.', '-']) * cycler('color', ['r', 'g', 'b', 'm', 'k'])
plt.rc('axes', prop_cycle=cc)

plt.rc('text', usetex=True)
plt.rc('font', family='Helvetica')

##

def solver(params):

    # ---------------------------------
    # Write gsi.xml in mixture.xml files

    with open('/Users/anabel/Documents/PhD/Code/Mutationpp_Michele/Mutationpp/data/mechanisms/air5_Park' +'.xml', 'r') as f:
        lines = f.readlines()

    # Replace the lines
    lines[12] = ('{:s}  \n').format('<arrhenius A="'+str(params[0])+'" n="-1.6" T="113200.0" />')  
    lines[18] = ('{:s}  \n').format('<arrhenius A="'+str(params[1])+'" n="-1.5" T="59360.0" />')  
    lines[24] = ('{:s}  \n').format('<arrhenius A="'+str(params[2])+'" n="0.0" T="75500.0" />')  
    lines[30] = ('{:s}  \n').format('<arrhenius A="'+str(params[3])+'" n="0.42" T="42938.0" />')  
    lines[35] = ('{:s}  \n').format('<arrhenius A="'+str(params[4])+'" n="0.0" T="19400.0" />')  
    

    # Write the lines back
    with open('/Users/anabel/Documents/PhD/Code/Mutationpp_Michele/Mutationpp/data/mechanisms/air5_Park' +'.xml', 'w') as f:
        f.writelines(lines)

    # ## Physico-chemical model settings (mpp)
    opts = mpp.MixtureOptions("air_5")
    opts.setThermodynamicDatabase("RRHO")
    opts.setStateModel("ChemNonEq1T")

    mix = mpp.Mixture(opts)
    ns = mix.nSpecies()
    nr = mix.nReactions()

    # ##

    T = 300
    P = 10000
    mix.equilibrate(T,P)
    rhoi_eq = mix.densities()

    Tinit = 15000
    mix.setState(rhoi_eq,Tinit,1)
    total_energy = mix.mixtureEnergyMass()*mix.density()

    time = 0.
    time_final = 1e-7
    dt = 1.e-9
    rhoi = rhoi_eq

    y0 = np.divide(rhoi_eq,mix.density())
    x0 = mix.convert_y_to_x(y0)
    temperature = np.array(mix.T())
    temp = [0]*1002
    # mass_fractions = np.array([y0])
    # mole_fraction =  np.array([x0])
    # time_reaction = np.array(time)

    while (time < time_final):
        dt = min(dt * 1.0002, 1.e-10)
        drhoi = runge_kutta4(mix, rhoi, total_energy, dt)
        rhoi += drhoi
        time += dt
        mix.setState(rhoi,total_energy,0)
        y = mix.Y()
        x = mix.convert_y_to_x(np.array(y))
        # mass_fractions = np.vstack((mass_fractions,[y]))
        # mole_fraction =  np.vstack((mole_fraction,[x]))
        temperature = np.vstack((temperature,mix.T()))
        # time_reaction  = np.vstack((time_reaction,time))

    for i in range(len(temp)):
        temp[i] += temperature[i][0]
    return temp

time_steps = 1002
time = [0]*time_steps
time_final = 1.e-7
for i in range(time_steps):
    time[i] += (i+1)*np.divide(time_final,time_steps)

## Input distributions ##
hyp = [[20.,22.],[20.,22.],[10.,15.],[10.,15.],[10.,15.]]
input_distr = dist.Uniform(5,hyp)

SA = sbl.Sobol(5,hyp) # Instantiation of sensitivity analysis object

sampls = SA.sampling_sequence(10000,5,input_distr,None)
print('Will compute '+str(len(sampls))+' solutions to the RK4 system')

f = np.zeros((len(sampls),time_steps))

for i in range(len(sampls)):
    params = [np.power(10,sampls[i][0]),np.power(10,sampls[i][1]),np.power(10,sampls[i][2]),np.power(10,sampls[i][3]),np.power(10,sampls[i][4])]
    f[i] = solver(params)
    print('Computed solution for sample number '+str(i))

ind = np.zeros((time_steps,5))
ind_T = np.zeros((time_steps,5))

ind_all = [[0.]*2]

for i in range(time_steps):
    func = [[0.]]*len(sampls)
    for j in range(len(sampls)):
        func[j] = [f[j,i]]

    ind[i] = SA.indices(func,10000,5)[0]
    ind_T[i] = SA.indices(func,10000,5)[1]
    print('Computed index for time step = '+str(i))

labels_T = {0: 'N2+M=2N+M', 1: 'O2+M=2O+M', 2: 'NO+M=N+O+M', 3: 'N2+O=NO+N', 4: 'NO+O=O2+N'}

plt.figure()
for i in range(5):
    plt.plot(time,ind_T[:,i],label=labels_T[i])

for i in range(4):
    plt.plot(time,ind[:,i])

plt.plot(time,ind[:,4],label='First order')

ax = plt.gca()
ax.yaxis.set_label_coords(0.0, 1.00)
plt.ylabel('Sobol indices', rotation=0)
plt.xlabel('time, s')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False) 
plt.ylim(0.,1.25)
plt.legend() 
plt.show()