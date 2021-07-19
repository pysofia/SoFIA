import sofia.Sobol as sbl
import numpy as np

def ishigami (x,y,z,a,b):
    return np.sin(x) + a*np.power(np.sin(y),2)+b*np.power(z,4)*np.sin(x)

def denorm(sample):
    return -np.pi+(np.pi*2*sample)

SA = sbl.Sobol() # Instantiation of sensitivity analysis object

sampls = SA.sampling_sequence(10000,3,['uniform','uniform','uniform'],None)

f = np.zeros((len(sampls),1))
for i in range(len(sampls)):
    f[i] += ishigami(denorm(sampls[i][0]),denorm(sampls[i][1]),denorm(sampls[i][2]),7.,0.1)

print(SA.indices(f,10000,3))