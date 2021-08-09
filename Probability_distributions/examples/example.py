import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sofia.distributions as dist

# 1D Uniform
hyp = [[-8.,8.]]

D = dist.Uniform(1,hyp)

x = np.linspace(-10.,10.,1000)

plt.plot(x,D.get_cdf_values(x))
plt.title('Uniform CDF')
plt.show()

sns.kdeplot(D.get_samples(10000),shade=True)
plt.title('Uniform PDF')
plt.show()

# 1D Gaussian
hyp = [[-1.,4.]]

D = dist.Gaussian(1,hyp)

x = np.linspace(-10.,10.,1000)

plt.plot(x,D.get_cdf_values(x))
plt.title('Gaussian CDF')
plt.show()

sns.kdeplot(D.get_samples(10000),shade=True)
plt.title('Gaussian PDF')
plt.show()

# 2D uncorrelated Gaussian
hyp = [[-1.,4.],[0.,2.]]

D = dist.Gaussian(2,hyp)

x = np.linspace(-10.,10.,1000)

plt.plot(x,D.get_cdf_values(x))
plt.plot(x,D.get_cdf_values(x,1))
plt.title('2D Gaussian CDF')
plt.show()

sns.kdeplot(D.get_samples(10000),shade=True)
sns.kdeplot(D.get_samples(10000,1),shade=True)
plt.title('2D Gaussian PDF')
plt.show()

plt.scatter(D.get_samples(1000),D.get_samples(1000,1))
plt.title('Samples 2D Gaussian')
plt.show()

# 2D uncorrelated Uniform
hyp = [[-1.,4.],[0.,2.]]

D = dist.Uniform(2,hyp)

x = np.linspace(-10.,10.,1000)

plt.plot(x,D.get_cdf_values(x))
plt.plot(x,D.get_cdf_values(x,1))
plt.title('2D uniform CDF')
plt.show()

sns.kdeplot(D.get_samples(10000),shade=True)
sns.kdeplot(D.get_samples(10000,1),shade=True)
plt.title('2D uniform PDF')
plt.show()

plt.scatter(D.get_samples(1000),D.get_samples(1000,1))
plt.title('Samples 2D uniform')
plt.show()