import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys

file = "./chain_"+sys.argv[1]+".gnu"

marginal = np.loadtxt(file)

sns.kdeplot(np.log10(marginal[:,0]),label="Qz")
sns.kdeplot(np.log10(marginal[:,1]),label="Cu")
sns.kdeplot(np.log10(marginal[:,2]),label="TPS")
plt.legend()
plt.show()