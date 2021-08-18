import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
import sofia.Sobol as sbl
import sofia.distributions as dist

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

N=1000
d=5
## Input distributions ##
hyp = [[20.,22.],[20.,22.],[10.,15.],[10.,15.],[10.,15.]]
input_distr = dist.Uniform(5,hyp)

SA = sbl.Sobol(5,hyp) # Instantiation of sensitivity analysis object

time_steps = 1002
time = [0]*time_steps
time_final = 1.e-7
for i in range(time_steps):
    time[i] += (i+1)*np.divide(time_final,time_steps)

ind = np.zeros((time_steps,5))
ind_T = np.zeros((time_steps,5))

ind_all = [[0.]*2]

####
for i in range(time_steps):
    func = np.zeros((N*(d+2),time_steps))

    # func = [[0.]]*N*(d+2)
    for j in range(N*(d+2)):
        func[j] = np.loadtxt('./output/temp_profile_'+str(j)+'.dat')
    print(func)
    exit(0)
####

for i in range(time_steps):
    func = [[0.]]*N*(d+2)
    for j in range(N*(d+2)):
        func[j] = np.loadtxt('./output/temp_profile_'+str(j)+'.dat')
    print(func)
    exit(0)
    ind[i] = SA.indices(func[:][i],N,5)[0]
    ind_T[i] = SA.indices(func[:][i],N,5)[1]
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