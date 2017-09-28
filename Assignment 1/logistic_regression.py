import numpy as np
import matplotlib.pyplot as plt
from math import log
from numpy.linalg import inv

def newton_method(X, y, theta, tolerance=1e-5, max_iters=15):
	MAX_ITERS = 1
	epoch = 1
	for _ in range(max_iters):
		H = hessian(X, theta)
		# print(hes)
		g = gradient(X, y, theta)
		# print('Gradient shape: ', g.shape)
		temp = theta - np.dot(inv(H), g)
		if np.sum(abs(theta - temp)) < tolerance:
			print('Convergered at epoch %d' % epoch)
			break
		theta = temp
		epoch += 1
	return theta

def gradient(X, y, theta):
	m = X.shape[0]
	h = hypothesis(X, theta)
	g = (1.0/m) * X.T.dot(h-y)
	return g

def hessian(X, theta):
	m = X.shape[0]
	h = hypothesis(X, theta)
	h.shape = (len(h),)
	H = (1.0/m) * np.dot(np.dot(X.T, np.diag(h)), np.dot(np.diag(1-h), X))
	return H

def sigmoid(z):
	result = 1.0/(1.0+np.exp(-z))
	return result

def cost_function(X, y, theta):
	m = X.shape[0]
	h = hypothesis(X, theta)
	J = (1.0/m) * (-y.dot(np.log(h)) - (1-y).dot(np.log(1-h)))
	return J

def hypothesis(X, theta):
	# print('Hypothesis: ', h.shape)
	return sigmoid(X.dot(theta))

def main():
	X = np.loadtxt('data/cx.dat')
	y = np.loadtxt('data/cy.dat')

	# Get positive and negative indices
	pos = np.nonzero(y)
	neg = np.where(y==0)

	# Plot
	plt.scatter(X[pos,0], X[pos,1], color='b', marker='+')
	plt.scatter(X[neg,0], X[neg,1], color='r', marker='o')
  	# plt.show()
  	
  	m = X.shape[0]
	X = np.column_stack((np.ones((m, 1)), X))
	y.shape = (y.shape[0], 1)
	theta = np.zeros((X.shape[1], 1))
	
	theta = newton_method(X, y, theta)
	print(theta)
	
	# predict if the student got 20 in exam 1 and 80 in exam 2
	test_data = np.array([1, 20, 80])
	p = 1 - hypothesis(test_data, theta)
	print('Probability that student is not admitted with a score of 20 on Exam 1 and a score of 80 on Exam 2 is %f' % p)
	
	# Plot decision boundary
	min_X = np.min(X[:,1:3])
	max_X = np.max(X[:,1:3])
	plot_x = [min_X-2, max_X+2]
	plot_y = (-1/theta[2])*(theta[1]*plot_x) + theta[0]
	plt.plot(plot_x, plot_y)
	plt.show()
main()