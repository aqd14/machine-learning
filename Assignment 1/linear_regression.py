# Implmentation of linear regession algorithm for only one feature dataset
# by using gradient descent to find optimum weights to the classification

'''
The data files ax.dat and ay.dat contain some example measurements of
heights for various boys between the ages of two and eights. The y-values
are the heights measured in meters, and the x-values are the ages of the boys
corresponding to the heights.

There are 50 training examples in total
'''

# usage python linear_regression.py

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import inv

class LinearRegression(object):
	""" Linear regression model

	Parameters
	----------
	alpha : float 
		Learning rate
		
	iter : integer 
		Number of iteration
		
	check_conv: boolean 
		Check if model is check_conv or not to stop iterations
	"""

	def __init__(self, alpha=0.01, iter=2000, check_conv=True):
		self.alpha = alpha
		self.iter = iter
		self.check_conv = check_conv
		# self.normalized = normalized

	def fit(self, X, y):
		""" Fit traning dataset to the model.
		Update theta simultaneously by using gradient descent.
		For simplicity, ignore error handling.


		X : integer
			training input
		y :	integer 
			label output
		"""

		self.theta = np.zeros((X.shape[1], 1)) # Column vector
		err_threshold = 1e-5 # Stop when the cost function is smaller than threshold
		J = []
		for i in range(self.iter):
			cost = cost_function(X, y, self.theta)
			J.append(cost)

			gra, _ = self.gradient(X, y)
			temp = self.theta - self.alpha * gra
			# if np.array_equal(temp, self.theta):
			if self.check_conv is True and np.sum(abs(self.theta - temp)) < err_threshold:
				break
			# Simultaneously update theta
			self.theta = temp

		# print('Not convergered!')
		return self.theta, J

	def hypothesis(self, X):
		# return np.dot(X, self.theta[1:]) + self.theta[0]
		return np.dot(X, self.theta)

	def gradient(self, X, y):
		"""Calculate gradient descent
		
		Parameters
		----------
		X : array-like
			training data
			
		Y : vector
			label output
			
		Returns
		-------
		gradient : float
			gradient value of given data
			
		mse : float
			mean squared errors
		"""
		err = y - self.hypothesis(X)
		mse = (1.0/X.shape[0]) * np.sum(np.power(err, 2))

		gradient = -(1.0/X.shape[0]) * X.T.dot(err)
		return gradient, mse

	def predict(self, X):
		return self.hypothesis(X)

def cost_function(X, y, theta):
	""" 
	Calculate cost function
	"""
	m = X.shape[0] # Training size
	err = y - np.dot(X, theta)
	J = 1.0/(2*m) * np.sum(np.power(err, 2))

	return J

# ----------------------- PART A ----------------------- #
def part_a():

	# load data
	X = np.loadtxt('data/ax.dat') # input data
	y = np.loadtxt('data/ay.dat') # output data

	# plot dataset
	plt.scatter(X, y, facecolors='blue')
	plt.xlabel('Age in years')
	plt.ylabel('Height in meters')
	# plt.show()

	# Data preprocessing
	m = X.shape[0] # Training size
	X = np.stack((np.ones(m), X), axis=-1)
	y = y.reshape(m, 1)

	model = LinearRegression(alpha=0.07)
	theta, J = model.fit(X, y)

	# Plot straight line
	plt.plot(X[:,1],np.dot(X,theta))
	plt.legend(['Linear Regression', 'Training Data'])
	plt.show()
	# plt.savefig('plot1.png')

	# Prediction
	test_data = np.array([[1, 3.5], [1, 7]])
	prediction = model.predict(test_data)
	print("Predicted height of kids age 3.5 and 7: ", prediction)
	
	plot_surface_contour(X, y)

def plot_surface_contour(X, y):
	# Display Surface Plot of J
	t0 = np.linspace(-3, 3, 100).reshape(100, 1)
	t1 = np.linspace(-1, 1, 100).reshape(100, 1)

	T0, T1 = np.meshgrid(t0, t1)

	J_vals = np.zeros((len(t0), len(t1)))

	for i in range(len(t0)):
		for j in range(len(t1)):
			t = np.hstack([t0[i], t1[j]])
			J_vals[i, j] = cost_function(X, y, t)

	#Because of the way meshgrids work with plotting surfaces
	#we need to transpose J to show it correctly
	J_vals = J_vals.T

	# print(J_vals)

	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.plot_surface(T0,T1,J_vals)
	plt.show()
	plt.close()
	#Display Contour Plot of J
	plt.contour(T0,T1,J_vals, np.logspace(-2,2,15))
	plt.show()


# ----------------------- END PART A --------------------- #


# ----------------------- PART B ------------------------- #
def part_b():
	# Load data from files
	X = np.loadtxt('data/bx.dat')
	y = np.loadtxt('data/by.dat')

	sigma = np.std(X, axis=0) # std
	mu = np.mean(X, axis=0) # mean

	# Data preprocessing
	m = X.shape[0]
	y = y.reshape(m, 1)

	
	X_scaled = (X-mu) / sigma # normalize(X)
	intercept = np.ones((m, 1))
	X_scaled = np.column_stack((intercept, X_scaled))
	# print(X_scaled)
	# y_scaled = normalize(y)

	iterations = 50

	alphas = [0.01, 0.03, 0.1, 0.3]
	colors = ['b', 'r', 'g', 'y']

	test_data = np.array([[1, 1650], [1, 3]])

	for i in range(len(alphas)):
		model = LinearRegression(alpha=alphas[i], iter=iterations, check_conv=False)
		theta, J = model.fit(X_scaled, y)
		
		# print('Learning rate %f' % alphas[i])
		# print('Theta: %d -- %d' % (theta[0], theta[1]))
		# print('Predict for the price of a house with 1650 square feet and 3 bedrooms: ', model.predict(test_data))
		
		# Now plot J
		plt.plot(range(iterations), J, color=colors[i], label=str(alphas[i]))

	plt.xlabel('Number of iterations')
	plt.ylabel('Cost J')
	plt.legend(loc='upper right')
	plt.show()

	# Find optimal theta vector with best learning rate
	# Assume we already know that alpha = 0.3 is the best learning rate
	model = LinearRegression(alpha=0.3, check_conv=True)
	theta, _ = model.fit(X_scaled, y)
	# print(theta)

	test_data = np.array([1650, 3])
	test_data_scaled = (test_data - mu) / sigma
	# print(test_data_scaled)
	test_data_scaled = np.append([1], test_data_scaled)
	# print(test_data_scaled)
	print('Predicted price using gradient descent: ', model.predict(test_data_scaled))

	# test_data_scaled.shape = (3, 1)
	# print(theta.T.dot(test_data_scaled))

	X = np.column_stack((intercept, X))
	test_data = np.append([1], test_data)
	theta_normal_equation = normal_equation(X, y)
	print('Predicted price using normal equation: ', theta_normal_equation.T.dot(test_data))

def normalize(X):
	# Preprocess data to give std of 1 and mean of 0
	sigma = np.std(X, axis=0) # std
	mu = np.mean(X, axis=0) # mean
	X = (X-mu) / sigma # adjustment
	return X

def normal_equation(X, y):
	theta = inv(X.T.dot(X)).dot(X.T).dot(y)
	return theta

# ----------------------- END PART B ------------------------- #

part_a()
part_b()
