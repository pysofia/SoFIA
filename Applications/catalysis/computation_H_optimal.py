import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import sys

json_file = "./models/models.json"
with open(json_file) as jfile:
    json_model = json.load(jfile)

file = "./output/chain_"+sys.argv[1]+".dat"

model = pickle.load(open(json_model[sys.argv[1]]["model"]["enthalpy"], 'rb'))
data = np.loadtxt(file)

H = [0.]*len(data[:,0])
for i in range(len(data[:,0])):
    H[i] = np.divide(model.mean([float(data[i,0]),float(data[i,1]),float(data[i,2])]),1e06)

plt.figure()
sns.kdeplot(H,shade=True)
plt.xlabel("$H^{\mathrm{opt}}$ [MJ/kg]")
plt.ylabel("$\mathcal{P}(H^{\mathrm{opt}})$")
plt.savefig("./Figures/H_MTAt1.png")
plt.show()