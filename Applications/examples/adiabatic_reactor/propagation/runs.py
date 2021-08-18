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

    shutil.copy('/Users/anabel/Documents/PhD/Code/Mutationpp_Michele/Mutationpp/data/mechanisms/air5_Park' +'.xml',
    '/Users/anabel/Documents/PhD/Code/Mutationpp_Michele/Mutationpp/data/mechanisms/air5_Park_'+str(params[5])+'.xml')

    shutil.copy('/Users/anabel/Documents/PhD/Code/Mutationpp_Michele/Mutationpp/data/mixtures/air_5' +'.xml',
    '/Users/anabel/Documents/PhD/Code/Mutationpp_Michele/Mutationpp/data/mixtures/air_5_'+str(params[5])+'.xml')

    with open('/Users/anabel/Documents/PhD/Code/Mutationpp_Michele/Mutationpp/data/mechanisms/air5_Park_'+str(params[5])+'.xml', 'r') as f:
        lines = f.readlines()

    # Replace the lines
    lines[12] = ('{:s}  \n').format('<arrhenius A="'+str(params[0])+'" n="-1.6" T="113200.0" />')  
    lines[18] = ('{:s}  \n').format('<arrhenius A="'+str(params[1])+'" n="-1.5" T="59360.0" />')  
    lines[24] = ('{:s}  \n').format('<arrhenius A="'+str(params[2])+'" n="0.0" T="75500.0" />')  
    lines[30] = ('{:s}  \n').format('<arrhenius A="'+str(params[3])+'" n="0.42" T="42938.0" />')  
    lines[35] = ('{:s}  \n').format('<arrhenius A="'+str(params[4])+'" n="0.0" T="19400.0" />')  
    

    # Write the lines back
    with open('/Users/anabel/Documents/PhD/Code/Mutationpp_Michele/Mutationpp/data/mechanisms/air5_Park_'+str(params[5]) +'.xml', 'w') as f:
        f.writelines(lines)

    with open('/Users/anabel/Documents/PhD/Code/Mutationpp_Michele/Mutationpp/data/mixtures/air_5_'+str(params[5])+'.xml', 'r') as g:
        lines = g.readlines()

    # Replace the lines
    lines[1] = ('{:s}  \n').format('<mixture mechanism="air5_Park_'+str(params[5])+'">') 

    # Write the lines back
    with open('/Users/anabel/Documents/PhD/Code/Mutationpp_Michele/Mutationpp/data/mixtures/air_5_'+str(params[5]) +'.xml', 'w') as g:
        g.writelines(lines)

    # ## Physico-chemical model settings (mpp)
    opts = mpp.MixtureOptions("air_5_"+str(params[5]))
    opts.setThermodynamicDatabase("RRHO")
    opts.setStateModel("ChemNonEq1T")

    mix = mpp.Mixture(opts)
    ns = mix.nSpecies()
    nr = mix.nReactions()

    # ##

    T = 300
    P = 100000
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

    # Write the lines back
    with open('/Users/anabel/Documents/PhD/Code/SoFIA/Applications/examples/adiabatic_reactor/propagation/output/temp_profile_p1atm_'+str(params[5])+'.dat', 'w') as f:
        for i in range(len(temp)):
            f.write(str(temp[i])+' ')
        f.write('\n')
    # return temp

    os.remove('/Users/anabel/Documents/PhD/Code/Mutationpp_Michele/Mutationpp/data/mechanisms/air5_Park_'+str(params[5])+'.xml')
    os.remove('/Users/anabel/Documents/PhD/Code/Mutationpp_Michele/Mutationpp/data/mixtures/air_5_'+str(params[5])+'.xml')
    return

def parallel_cases_with_mpi(cases, run_func, verbose=True):
    """
    Run a list of "cases" in parallel, divided accross all available cores.

    Specifically, this function calls run_func for each of the case descriptions
    in cases, equally divided accross the available CPU cores.  When more than
    one core is available, the order of the cases run is not guaranteed.  For
    a single core, cases are run in order.

    Note: if the runtime for each job varies significantly, you can shuffle the
    order of the cases list to improve load balancing.

    Parameters
    ==========
    cases: list
        objects representing parameterizations defining cases to run
    run_func: function object
        function which runs a given parameterization (item in cases)
    verbose: bool
        if True, information is printed regarding which core runs which case,
        otherwise nothing is printed
    """
    if 'mpi4py' not in sys.modules: from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    info = f'[{rank}]: ' if size > 1 else ''

    if verbose:
        if rank == 0 and size > 1:
            print(f'Running {len(cases)} cases with {size} processors.')
            print(f'MPI Version: {MPI.Get_version()}', flush=True)
        comm.Barrier()

    for i in range(len(cases)):
        case_num = i * size + rank

        if case_num < len(cases):
            case = cases[case_num]
            if verbose:
                print(info + f'running case {case_num} - {case}', flush=True)
            run_func(case)
        else:
            break

time_steps = 1002
time = [0]*time_steps
time_final = 1.e-7
for i in range(time_steps):
    time[i] += (i+1)*np.divide(time_final,time_steps)

## Input distributions ##
hyp = [[20.,22.],[20.,22.],[10.,15.],[10.,15.],[10.,15.]]
input_distr = dist.Uniform(5,hyp)

sampls = [[input_distr.get_one_sample(),input_distr.get_one_sample(pos=1),input_distr.get_one_sample(pos=2),input_distr.get_one_sample(pos=3),input_distr.get_one_sample(pos=4)] for i in range(1000)]

f = np.zeros((len(sampls),time_steps))
params = [0.]*len(sampls)

for i in range(len(sampls)):
    params[i] = [np.power(10,sampls[i][j]) for j in range(5)]
    params[i].append(i)

# Run the cases
parallel_cases_with_mpi(params, solver)