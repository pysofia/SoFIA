import pickle
import numpy as np
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, RBF
import sofia.GPR as Regressor

## Loading training data ##
realizations_dir = '/Users/anabel/Documents/PhD/Code/stagline/Nitridation_2T/2T_SEB_wrecombination/realizations_G67_incomplete.dat'
data_dir = '/Users/anabel/Documents/PhD/Code/stagline/Nitridation_2T/2T_SEB_wrecombination/points_incomplete.dat'

X = np.loadtxt(data_dir)
Y = np.loadtxt(realizations_dir)

X = np.asarray(X)
Y = np.asarray(Y)

## with my library ##
h=[1.,1.,1.5,1.]
model = Regressor.GP(h)
model.set_data(X,Y[:,0])
model.train()

# save the model to disk
f = './models/rec/rec_2T_SEB_mylibrary.sav'
pickle.dump(model,open(f,"wb"))