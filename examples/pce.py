import sofia.pc as pce
import numpy as np
import matplotlib.pyplot as plt
import sofia.distributions as dist

"""

        Sampling a uniform distribution with Gaussians using PCE

"""

# Generate Gaussian samples to use
N = 5000
hyp=[[0.,1.]]
G = dist.Gaussian(1,hyp)
S = G.get_samples(N)

hypU=[[0.,1.]]
target = dist.Uniform(1,hypU)

h = target.fun_icdf()

ki_uniform = pce.approximate_rv_coeffs(13, h)
k = pce.generate_rv(ki_uniform, S)

plt.hist(k, bins=50)
plt.show()