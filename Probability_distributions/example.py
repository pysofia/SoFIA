import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import distributions as dist

# 1D Uniform
hyp = [[-8.,8.]]

D = dist.Uniform(1,hyp)

x = np.linspace(-10.,10.,1000)

plt.plot(x,D.get_cdf_values(x,0))
plt.show()

sns.kdeplot(D.get_samples(0,10000),shade=True)
plt.show()

# 1D Gaussian
hyp = [[-1.,4.]]

D = dist.Gaussian(1,hyp)

x = np.linspace(-10.,10.,1000)

plt.plot(x,D.get_cdf_values(x,0))
plt.show()

sns.kdeplot(D.get_samples(0,10000),shade=True)
plt.show()

# 2D uncorrelated Gaussian
hyp = [[-1.,4.],[0.,2.]]

D = dist.Gaussian(2,hyp)

x = np.linspace(-10.,10.,1000)

plt.plot(x,D.get_cdf_values(x,0))
plt.plot(x,D.get_cdf_values(x,1))
plt.show()

sns.kdeplot(D.get_samples(0,10000),shade=True)
sns.kdeplot(D.get_samples(1,10000),shade=True)
plt.show()

plt.scatter(D.get_samples(0,1000),D.get_samples(1,1000))
plt.show()

# 2D uncorrelated Uniform
hyp = [[-1.,4.],[0.,2.]]

D = dist.Uniform(2,hyp)

x = np.linspace(-10.,10.,1000)

plt.plot(x,D.get_cdf_values(x,0))
plt.plot(x,D.get_cdf_values(x,1))
plt.show()

sns.kdeplot(D.get_samples(0,10000),shade=True)
sns.kdeplot(D.get_samples(1,10000),shade=True)
plt.show()

plt.scatter(D.get_samples(0,1000),D.get_samples(1,1000))
plt.show()