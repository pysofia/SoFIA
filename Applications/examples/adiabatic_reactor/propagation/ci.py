import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

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

N = 1000
time_steps = 1002

time = [0]*time_steps
time_final = 1.e-7
for i in range(time_steps):
    time[i] += (i+1)*np.divide(time_final,time_steps)

func = np.zeros((N,time_steps))
func2 = np.zeros((N,time_steps))

for j in range(N):
    func[j] = np.loadtxt('./output/temp_profile_'+str(j)+'.dat')
    func2[j] = np.loadtxt('./output/temp_profile_p15_'+str(j)+'.dat')

## 95 % C.I. ##

mean = [0.]*time_steps
lower = [0.]*time_steps
upper = [0.]*time_steps

mean2 = [0.]*time_steps
lower2 = [0.]*time_steps
upper2 = [0.]*time_steps

for i in range(time_steps):
    m = [0.]*N
    m2 = [0.]*N
    for j in range(N):
        m[j] = func[j][i]
        m2[j] = func2[j][i]

    mean[i] = np.mean(m)
    upper[i] = np.percentile(m,97.5)
    lower[i] = np.percentile(m,2.5)

    mean2[i] = np.mean(m2)
    upper2[i] = np.percentile(m2,97.5)
    lower2[i] = np.percentile(m2,2.5)

labels_T = {0: 'N2+M=2N+M', 1: 'O2+M=2O+M', 2: 'NO+M=N+O+M', 3: 'N2+O=NO+N', 4: 'NO+O=O2+N'}
colors = {0: 'green', 1: 'red', 2: 'blue', 3: 'orange', 4: 'pink'}

plt.figure()

plt.plot(time,mean,label='100hPa',color=colors[0])
plt.plot(time,upper,linestyle='--',color=colors[0])
plt.plot(time,lower,linestyle='--',color=colors[0])
plt.fill_between(time,lower,upper,color=colors[0],alpha=0.2,label='95\% C.I.')

plt.plot(time,mean2,label='15hPa',color=colors[1])
plt.plot(time,upper2,linestyle='--',color=colors[1])
plt.plot(time,lower2,linestyle='--',color=colors[1])
plt.fill_between(time,lower2,upper2,color=colors[1],alpha=0.2,label='95\% C.I.')

ax = plt.gca()
ax.yaxis.set_label_coords(0.0, 1.00)
plt.ylabel('Temperature, K', rotation=0)
plt.xlabel('time, s')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False) 
plt.xlim(0.,np.power(10.,-7.))
# plt.ylim(6000., 15500.)
plt.legend() 
plt.show()
