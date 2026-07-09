import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from neural_network.layers import DenseLayer
from neural_network.activation_functions import tanh, tanh_deriv
from neural_network.training import train_network
from neural_network.loss_functions import mse, gradient_mse_a_L

random_seed = 11
np.random.seed(random_seed)

x, y = fetch_california_housing(return_X_y=True)

avg_rooms_per_occup = x[:, 2] / x[:, 5]
x = np.column_stack((x, avg_rooms_per_occup))

train_x, temp_x, train_y, temp_y = train_test_split(x, y, test_size=0.3, random_state=random_seed)
valid_x, test_x, valid_y, test_y = train_test_split(temp_x, temp_y, test_size=0.5, random_state=random_seed)

scaler = StandardScaler()

train_x = scaler.fit_transform(train_x)
valid_x = scaler.transform(valid_x)
test_x = scaler.transform(test_x)

network = [
	DenseLayer(9, 128, tanh, tanh_deriv),
	DenseLayer(128, 128, tanh, tanh_deriv),
	DenseLayer(128, 1, lambda x : x, lambda x : 1),
]

train_network(
    network,
    train_x,
    train_y,
    16,
    mse,
    gradient_mse_a_L,
    200,
    0.003,
    0.9999,
    valid_x,
    valid_y)

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