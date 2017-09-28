'''
Created on Sep 19, 2017

@author: aqd14
'''

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv, norm
from map_features import *

def newton_method(X, y, _lambda, tolerance=1e-5, max_iters=20):
    theta = np.zeros((X.shape[1], 1))
    epoch = 1
    for _ in range(max_iters):
        H = regularized_hessian(X, theta, _lambda)
        # print(hes)
        g = regularized_gradient(X, y, theta, _lambda)
        # print('Gradient shape: ', g.shape)
        temp = theta - np.dot(inv(H), g)
        if np.sum(abs(theta - temp)) < tolerance:
            print('Convergered at epoch %d' % epoch)
            break
        theta = temp
        epoch += 1
    if epoch >= max_iters:
        print('Reached maximum iteration!')
    return theta

def regularized_gradient(X, y, theta, _lambda):
    m = X.shape[0]
    h = hypothesis(X, theta)
    g = (1.0/m) * X.T.dot(h-y)
    # Adjust result with regularization parameter
    g[1:] += (_lambda * theta[1:])/m
    return g

def regularized_hessian(X, theta, _lambda):
    m = X.shape[0]
    h = hypothesis(X, theta)
    h.shape = (len(h),)
    H = (1.0/m) * np.dot(np.dot(X.T, np.diag(h)), np.dot(np.diag(1-h), X))
    # Adjust result with regularization parameter
    reg_diag_matrix = np.diag(np.ones(X.shape[1]))
    reg_diag_matrix[0][0] = 0
    H += (_lambda * reg_diag_matrix)/m
    return H

def sigmoid(z):
    result = 1.0/(1.0+np.exp(-z))
    return result

def regularized_cost_function(X, y, theta, _lambda):
    """Calculate cost function with sigmoid activation function and regularization
    
    Parameters
    ----------
    X : array-like
        Training input data
    y : array-like
        Training output data
    """
    m = X.shape[0]
    h = hypothesis(X, theta)
    J = (_lambda * theta[1:]**2)/(2*m) + (1.0/m) * (-y.dot(np.log(h)) - (1-y).dot(np.log(1-h)))
    return J

def hypothesis(X, theta):
    # print('Hypothesis: ', h.shape)
    return sigmoid(X.dot(theta))

def main():
    # Load dataset
    X = np.loadtxt('pa2data/bx.dat', delimiter=',')
    y = np.loadtxt('pa2data/by.dat')
    
    # Find indices of positive and negative examples
    pos = np.nonzero(y)[0]
    neg = np.where(y==0)[0]
    
    # Plot out the raw data
    '''
    plt.scatter(X[pos, 0], X[pos, 1], marker="+", color="b")
    plt.scatter(X[neg, 0], X[neg, 1], marker="o", color="r")
    plt.show()
    '''
    
    # Define the ranges of the grid
    u = np.linspace(-1, 1.5, 200)
    v = np.linspace(-1, 1.5, 200)
    
    # Reshape to be 2-D
    u.shape = (len(u), 1)
    v.shape = (len(v), 1)
    
    # Plotting
    X_axis, Y_axis = np.meshgrid(u, v)
    Z = np.zeros((len(u), len(v)))
    
    # Prepare data for Newton method
    # Create more features for our training data with feature mappings
    X_added_features = map_features(X[:, 0],X[:,1])
    # m = X.shape[0]
    # X = np.column_stack((np.ones((m, 1)), X))
    y.shape = (y.shape[0], 1)
    _lambdas = np.array([0, 1, 5, 10, 100, 1000000])
    
    test_data = map_features(np.array([0.5]), np.array([0.5]))
    for t in range(_lambdas.size):
        theta = newton_method(X_added_features, y, _lambdas[t])
        print('Theta value = %s\n' % theta[:,0])
        print('Lambda = %d \t L2-norm = %f\n' % (_lambdas[t], norm(theta)))
#         theta.shape = (1, theta.shape[0])
        print('Prediction value for input (0.5, 0.5) is %f' % hypothesis(test_data, theta))
        print('------------- END ---------------')
        for i in range(len(u)):
            for j in range(len(v)):
                Z[i][j] = np.dot(map_features(u[i], v[j]), theta)
                
        plt.clf()
        plt.scatter(X[pos,0], X[pos,1], marker='+', color='b')
        plt.scatter(X[neg,0], X[neg,1], marker='o', color='r')
        plt.axis('equal')
        plt.contour(X_axis, Y_axis, Z.T, 0, linewidth=2)
        plt.show()
        
if __name__ == '__main__':
    main()