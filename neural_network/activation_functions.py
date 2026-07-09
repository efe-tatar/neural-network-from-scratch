import numpy as np

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
	sig_x = sigmoid(x)
	return sig_x * (1 - sig_x)

def ReLU(x):
	return np.maximum(0, x)

def ReLU_deriv(x):
	return np.where(x > 0, 1, 0)

def tanh(x):
	return np.tanh(x)

def tanh_deriv(x):
	return 1 - np.tanh(x) ** 2