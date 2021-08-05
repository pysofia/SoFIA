import sofia.pc as pce
import numpy as np
import matplotlib.pyplot as plt

# Generate a bunch of Gaussian random variables to use
N = 5000
S = np.random.normal(loc=0,scale=1,size=N)

h = pce.expo_icdf([])

ki_uniform = pce.approximate_rv_coeffs(13, h)
k = pce.generate_rv(ki_uniform, S)

plt.hist(k, bins=50)
plt.show()