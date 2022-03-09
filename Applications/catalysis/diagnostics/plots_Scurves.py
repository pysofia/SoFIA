import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import sofia.data_assembly_cat as assemble
import json
import pickle
import os

class cd:
    """Context manager for changing the current working directory""" # Needed for the handling of models files
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

mpl.rcParams.update({
        'text.usetex' : True,
        'lines.linewidth' : 2,
        'axes.labelsize' : 40,
        'xtick.labelsize' : 40,
        'ytick.labelsize' : 40,
        'legend.fontsize' : 30,
        'font.family' : 'palatino',
        'font.size' : 40,
        'savefig.format' : 'png',
        'lines.linestyle' : '-',
        'xtick.major.pad' : 10
        })

## Posteriors ##

file = "../output/chain_"+sys.argv[1]+".dat"

sofia_dir = '/Users/anabel/Documents/PhD/Code/' # Directory wherever you have the SoFIA library
json_file = sofia_dir +'SoFIA/Applications/catalysis/models/models.json'

with open(json_file) as jfile:
    json_model = json.load(jfile)

with cd(sofia_dir + "SoFIA/Applications/catalysis"):
    H_model = pickle.load(open(json_model[sys.argv[1]]["model"]["enthalpy"], 'rb'))

chain = np.loadtxt(file)

with cd(sofia_dir + "SoFIA/Applications/catalysis"):
    assembly = assemble.assembly(json_file)
assembly.assembly_hyperparameters()

H = [0.]*len(chain[0:5000,0])
G = np.ndarray((len(H),3))
for i in range(len(H)):
    G[i] = assembly.denormalization([float(chain[i,0]),float(chain[i,1]),float(chain[i,2])])
    H[i] = H_model.mean([float(chain[i,0]),float(chain[i,1]),float(chain[i,2])])

G1 = G[:,0]
G2 = G[:,1]
G3 = G[:,2]

## Nominal

Qz = np.reshape(np.loadtxt('/Users/anabel/Documents/PhD/Stagiaires_DCs/Diana/Computations/MTAt1_3D_2/Scurve_Qz.out'),(-1, 9))
Cu = np.reshape(np.loadtxt('/Users/anabel/Documents/PhD/Stagiaires_DCs/Diana/Computations/MTAt1_3D_2/Scurve_Cu.out'),(-1, 9))
TP = np.reshape(np.loadtxt('/Users/anabel/Documents/PhD/Stagiaires_DCs/Diana/Computations/MTAt1_3D_2/Scurve_TP.out'),(-1, 9))


fig = plt.figure(figsize=(13, 11))
# set_rc_params()

plt.scatter(G2[0:5000], H[0:5000], color=CB_color_cycle[1],alpha=0.1)
plt.scatter(G3[0:5000], H[0:5000], color=CB_color_cycle[2],alpha=0.1)
plt.scatter(G1[0:5000], H[0:5000], color=CB_color_cycle[0],alpha=0.1)

plt.scatter(G2[0:1], H[0:1], color=CB_color_cycle[1],label='Cu')
plt.scatter(G3[0:1], H[0:1], color=CB_color_cycle[2],label='TPS')
plt.scatter(G1[0:1], H[0:1], color=CB_color_cycle[0],label='Qz')

plt.plot(np.log10(Cu[:,0]), Cu[:,2], color='orange')
plt.plot(np.log10(Qz[:,0]), Qz[:,2], color='blue')
plt.plot(np.log10(TP[:,0]), TP[:,2], color='green')

plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, borderaxespad=0.)

plt.xlim(-4.,0.)
plt.xlabel('Catalytic recombination')
plt.ylabel('Enthalpy [MJ/kg]') #plt.ylabel('$H_{\delta} \ \mathrm{[MJ/kg]}$')
labs=['$10^{-4}$','$10^{-3}$','$10^{-2}$','$10^{-1}$','$10^{0}$']
labs_y=['$5$','$10$','$20$','$30$','$40$','$50$']
plt.xticks([-4.,-3.,-2.,-1,0.],labs)
plt.yticks([5000000.,10000000.,20000000.,30000000.,40000000.,50000000.],labs_y)
plt.legend(loc='best')
plt.show()
# plt.savefig('/Users/anabel/Documents/PhD/Code/THESIS/Chapter_6/'+'Scurves_t2_black',transparent=True)
