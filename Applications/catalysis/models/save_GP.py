import pickle
import numpy as np
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, RBF
import sofia.GPR as Regressor

file = "/Users/anabel/Documents/PhD/Stagiaires_DCs/Diana/Computations/MTAt7_3D/output_MTAt7.out"

data = np.loadtxt(file)

gamma_norm_1 = (data[:,8]-np.min(data[:,8]))/(np.max(data[:,8])-np.min(data[:,8]))
gamma_norm_2 = (data[:,9]-np.min(data[:,9]))/(np.max(data[:,9])-np.min(data[:,9]))
gamma_norm_3 = (data[:,10]-np.min(data[:,10]))/(np.max(data[:,10])-np.min(data[:,10]))

lik = -data[:,7]

X = [0.]*len(gamma_norm_1)
Y = [0.]*len(gamma_norm_1)
for i in range(len(gamma_norm_1)):
    X[i] = [gamma_norm_1[i],gamma_norm_2[i],gamma_norm_3[i]]
    Y[i] = lik[i]

X = np.asarray(X)
Y=np.asarray(Y)

## with my library ##
h=[1.,1.,1.5,1.]
model = Regressor.GP(h)
model.set_data(X,Y)
model.train()

## with scikit-learn ##
# kernel = RBF()#Matern(length_scale=(0.5,0.5,0.5), nu=3/2)

# model = gaussian_process.GaussianProcessRegressor(kernel=kernel,normalize_y=False)
# model.fit(X, Y)

# save the model to disk
f = './MTAt7.sav'
pickle.dump(model,open(f,"wb"))