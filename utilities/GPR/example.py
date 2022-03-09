import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sofia.GPR as Regressor

def function_to_approximate(x):
    return x*np.sin(x)

X = np.random.rand(6)*10.
Y = function_to_approximate(X)

print('Training GP')

hyper = [1.,1.,1.5,1.]
model = Regressor.GP(hyper)

model.set_data(X,Y)
model.train()

X_test = np.linspace(0.,10.,100) 
y = [0.]*len(X_test)
var = [0.]*len(X_test)

for i in range(len(X_test)):
    y[i] = model.mean(X_test[i])
    var[i] = model.variance(X_test[i])

y_mean = [0.]*100
y_std = [0.]*100

y_p2 = [0.]*100
y_m2 = [0.]*100
for i in range(100):
    y_mean[i] += y[i]
    y_std[i] += np.sqrt(var[i])
    y_p2[i] += y_mean[i] + 2*y_std[i]
    y_m2[i] += y_mean[i] - 2*y_std[i]

mpl.rcParams.update({
        'text.usetex' : True,
        'lines.linewidth' : 5,
        'axes.labelsize' : 30,
        'xtick.labelsize' : 30,
        'ytick.labelsize' : 30,
        'legend.fontsize' : 25,
        'font.family' : 'palatino',
        'font.size' : 20,
        'savefig.format' : 'png',
        'lines.linestyle' : '-'
        })

fig = plt.figure(figsize=(12,8))

plt.plot(X_test,function_to_approximate(X_test),color='black',linestyle='--',label='$f(x) = x \sin(x)$')

plt.plot(X_test,y_mean,label='GP mean')
plt.fill_between(X_test,y_m2,y_p2,alpha=0.2,label='95\% C.I.')
plt.scatter(X,Y,edgecolors='red',color='black',s=280,linewidths=2,label='Observations')
plt.xlabel('$x$',fontsize = 20)
plt.ylabel('$f(x)$',fontsize = 20)
plt.xlim(0,10)

plt.show()