'''
Created on Sep 17, 2017

@author: doquocanh-macbook
'''

import numpy as np
from numpy.linalg import inv, norm
# from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

X = np.loadtxt('pa2data/ax.dat')
y = np.loadtxt('pa2data/ay.dat')

# print(X)
# print(y)
plt.scatter(X, y, facecolors='black')
# plt.show()

# number of training data
m = X.shape[0]

# increase model capacity by adding higher polynomial
X = np.stack((np.ones(m), X, X**2, X**3, X**4, X**5), axis=-1)

# number of feature
n = X.shape[1]
diagonal_matrix = np.diag(np.ones(n))
diagonal_matrix[0][0] = 0

_lambdas = np.array([0, 1, 5, 10, 100, 1000000])
thetas = []

for i in range(_lambdas.size):
    theta = inv(X.T.dot(X) + _lambdas[i] * diagonal_matrix)
    theta = theta.dot(X.T)
    theta = theta.dot(y)
    thetas.append(theta)
    
# calculate norm for thetas
for i in range(len(thetas)):
    print('Lambda = %d \t L2-norm = %f' % (_lambdas[i], norm(thetas[i])))
    
# input value range    
r = np.arange(-1,1,0.05)

features = np.stack((np.ones(r.shape[0]), r, r**2, r**3, r**4, r**5), axis=-1)
# plot data
colors = ['b', 'g', 'r', 'c', 'm', 'y']
for i in range(_lambdas.size):
    print(thetas[i])
    plt.plot(r, features.dot(thetas[i]), color=colors[i], label='lambda ' + str(_lambdas[i]))

plt.legend(loc='upper right')
plt.show()


