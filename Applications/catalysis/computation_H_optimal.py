import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import sys
from sofia.run_parallel import parallel_cases_with_mpi

json_file = "./models/models.json"
with open(json_file) as jfile:
    json_model = json.load(jfile)

file = "./output/chain_"+sys.argv[1]+".dat"

model = pickle.load(open(json_model[sys.argv[1]]["model"]["enthalpy"], 'rb'))
data = np.loadtxt(file)

if sys.argv[2]=="parallel":
    H = []
    # Parallel #

    def H_parallel(x,H):
        # H[x[-1]] = np.divide(model.mean(x[0:-1]),1e06)
        return np.divide(model.mean(x[0:-1]),1e06)

    cases = []
    for i in range(len(data[:,0])):
        cases.append([float(data[i,0]),float(data[i,1]),float(data[i,2]),i])
    
    # Run the cases
    H = parallel_cases_with_mpi(cases[0:12], H_parallel, H)
    print(H)
    exit(0)

    plt.figure()
    sns.kdeplot(H,shade=True)
    plt.xlabel("$H^{\mathrm{opt}}$ [MJ/kg]")
    plt.ylabel("$\mathcal{P}(H^{\mathrm{opt}})$")
    # plt.savefig("./Figures/H_MTAt1.png")
    plt.show()

else:
    H = [0.]*len(data[:,0])
    for i in range(len(data[:,0])):
        H[i] = np.divide(model.mean([float(data[i,0]),float(data[i,1]),float(data[i,2])]),1e06)

    plt.figure()
    sns.kdeplot(H,shade=True)
    plt.xlabel("$H^{\mathrm{opt}}$ [MJ/kg]")
    plt.ylabel("$\mathcal{P}(H^{\mathrm{opt}})$")
    plt.savefig("./Figures/H_MTAt4.png")
    plt.show()