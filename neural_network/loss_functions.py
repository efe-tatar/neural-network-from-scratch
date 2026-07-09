import numpy as np

def mse(y_pred:np.ndarray, y_true:np.ndarray):
	return np.mean((y_true - y_pred) ** 2)

def gradient_mse_a_L(y_pred:np.ndarray, y_true:np.ndarray):
	return - (2 / y_pred.size) * (y_true - y_pred)