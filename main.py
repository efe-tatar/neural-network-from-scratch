
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

random_seed = 11

np.random.seed(random_seed)

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

def mse(y_pred:np.ndarray, y_true:np.ndarray):
	return np.mean((y_true - y_pred) ** 2)

def gradient_mse_a_L(y_pred:np.ndarray, y_true:np.ndarray):
	return - (2 / y_pred.size) * (y_true - y_pred)

def delta_a_L_over_delta_z_L(a_L:np.ndarray, z_L:np.ndarray, deriv_act_func):
	return np.array([
		deriv_act_func(z_L[i]) for i in range(z_L.size)
	])

class DenseLayer:
	def __init__(self, input_size, output_size, act_func, derv_act_func):
		self.input_size = input_size
		self.output_size = output_size
		self.act_func = act_func
		self.derv_act_func = derv_act_func

		self.W = np.random.uniform(-1, 1, (output_size, input_size)) * (1 / np.sqrt(input_size))
		self.B = np.random.uniform(-0.1, 0.1, output_size)
	
	def forward(self, x:np.ndarray):
		self.z = self.W @ x + self.B
		self.a = self.act_func(self.z)
		return self.a


network = [
	DenseLayer(9, 128, tanh, tanh_deriv),
	DenseLayer(128, 128, tanh, tanh_deriv),
	DenseLayer(128, 1, lambda x : x, lambda x : 1),
]

feature_names = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]

x, y = fetch_california_housing(return_X_y=True)

avg_rooms_per_occup = x[:, 2] / x[:, 5]
x = np.column_stack((x, avg_rooms_per_occup))

train_x, temp_x, train_y, temp_y = train_test_split(x, y, test_size=0.3, random_state=random_seed)
valid_x, test_x, valid_y, test_y = train_test_split(temp_x, temp_y, test_size=0.5, random_state=random_seed)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

train_x = scaler.fit_transform(train_x)
valid_x = scaler.transform(valid_x)
test_x = scaler.transform(test_x)

epochs = 70
epoch_training_losses = []
epoch_validation_losses = []
base_learning_rate = 0.001
decay_rate = 0.97

display_every = 500

for epoch in range(epochs):

	lr = base_learning_rate * decay_rate ** epoch

	print(f"-- training epoch: {epoch+1} --")
	print(f"learning rate: {lr}")

	training_losses = []

	for training_index in range(len(train_x)):
		x = train_x[training_index]
		y = train_y[training_index]

		y_pred = x

		for l in network:
			y_pred = l.forward(y_pred)
		
		loss = mse(y_pred, y)
		training_losses.append(loss)

		last_delta = None

		grad_W = []
		grad_B = []
		
		for i in range(len(network)-1, -1, -1):

			if last_delta is None:
				delta_C_over_delta_aL = gradient_mse_a_L(y_pred, y)
				delta_aL_over_delta_zL = delta_a_L_over_delta_z_L(
					network[i].a,
					network[i].z,
					network[i].derv_act_func
				)
				delta_l = np.multiply(delta_C_over_delta_aL, delta_aL_over_delta_zL)
			else:
				#delta_l = network[i+1].W.T @ last_delta
				delta_l = (network[i+1].W.T @ last_delta) * network[i].derv_act_func(network[i].z)
			last_delta = delta_l
			a:np.ndarray = network[i-1].a if i > 0 else x
			gradient_W = np.outer(delta_l, a)
			gradient_B = delta_l

			grad_W.append(gradient_W)
			grad_B.append(gradient_B)
		
		grad_W.reverse()
		grad_B.reverse()
		for i in range(len(grad_W)):
			network[i].W = network[i].W - lr * grad_W[i]
			network[i].B = network[i].B - lr * grad_B[i]
		
	training_losses = np.array(training_losses)
	mean_training_loss = training_losses.mean()
	epoch_training_losses.append(mean_training_loss)

	validation_losses = []
	for validation_index in range(len(valid_x)):
		x = valid_x[validation_index]
		y = valid_y[validation_index]

		y_pred = x

		for l in network:
			y_pred = l.forward(y_pred)
		
		loss = mse(y_pred, y)
		validation_losses.append(loss)
	validation_losses = np.array(validation_losses)
	mean_validation_loss = validation_losses.mean()
	epoch_validation_losses.append(mean_validation_loss)
	print(f"mean training loss: {mean_training_loss}")
	print(f"mean validation loss: {mean_validation_loss}")

print("-- training complete --")

test_losses = []
for test_index in range(len(test_x)):
	x = test_x[test_index]
	y = test_y[test_index]

	y_pred = x

	for l in network:
		y_pred = l.forward(y_pred)
	
	loss = mse(y_pred, y)
	test_losses.append(loss)
test_losses = np.array(test_losses)
mean_test_loss = test_losses.mean()

print(f"-- mean test loss: {mean_test_loss} --")

import matplotlib.pyplot as plt

plt.plot(epoch_training_losses, color="blue", label="training loss")
plt.plot(epoch_validation_losses, color="red", label="validation loss")

plt.title("Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)

plt.show()