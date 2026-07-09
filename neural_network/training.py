import numpy as np
from neural_network.layers import DenseLayer

def train_network(
	network,
	x_train_set,
	y_train_set,
	batch_size,
	loss_function,
	loss_function_deriv,
	epochs,
	base_learning_rate,
	decay_rate,
	x_valid_set = None,
	y_valid_set = None,
	metrics = None,):

	training_set_loss_per_epoch = []
	validation_set_loss_per_epoch = []

	for epoch in range(epochs):
		print(f"-- training epoch: {epoch+1} --")

		lr = base_learning_rate * decay_rate ** epoch
		print(f"learning rate: {lr}")

		indices = np.random.permutation(len(x_train_set))
		shuffled_x_train_set = x_train_set[indices]
		shuffled_y_train_set = y_train_set[indices]

		x_train_batches = np.array_split(shuffled_x_train_set, np.arange(batch_size, len(shuffled_x_train_set), batch_size))
		y_train_batches = np.array_split(shuffled_y_train_set, np.arange(batch_size, len(shuffled_y_train_set), batch_size))
		#x_train_batches = np.split(x_train_set, batch_size)
		#y_train_batches = np.split(y_train_set, batch_size)

		mean_training_set_loss_current_epoch = 0
		mean_validation_set_loss_current_epoch = 0

		for batch_index in range(len(x_train_batches)):

			x_train_batch = x_train_batches[batch_index]
			y_train_batch = y_train_batches[batch_index]
			
			for instance_index in range(len(x_train_batch)):
				x = x_train_batch[instance_index]
				y_true = y_train_batch[instance_index]

				y_pred = x
				for layer in network:
					y_pred = layer.forward(y_pred)
				
				loss = loss_function(y_pred, y_true)
				mean_training_set_loss_current_epoch = mean_training_set_loss_current_epoch + loss

				last_delta = None
				grad_W = []
				grad_B = []

				for i in range(len(network)-1, -1, -1):

					if last_delta is None:
						# delta_l = dLoss/daL * daL/dzL
						# these are partial derivatives don't get confused
						delta_C_over_delta_aL = loss_function_deriv(y_pred, y_true)
						delta_aL_over_delta_zL = network[i].derv_act_func(network[i].z)

						delta_l = np.multiply(delta_C_over_delta_aL, delta_aL_over_delta_zL)

					else:
						# note: do not get confused by mixed use of row/col vectors
						# delta_l-1 = delta_l * dzL/daL-1 * daL-1/dzL-1
						# these are partial derivatives don't get confused
						delta_l = (network[i+1].W.T @ last_delta) * network[i].derv_act_func(network[i].z)

					last_delta = delta_l

					a:np.ndarray = network[i-1].a if i > 0 else x

					gradient_W = np.outer(delta_l, a)
					gradient_B = delta_l

					network[i].W_gradient = network[i].W_gradient + gradient_W
					network[i].B_gradient = network[i].B_gradient + gradient_B

					# grad_W.append(gradient_W)
					# grad_B.append(gradient_B)
				
				# grad_W.reverse()
				# grad_B.reverse()

				"""for i in range(len(grad_W)):
					network[i].W = network[i].W - lr * grad_W[i]
					network[i].B = network[i].B - lr * grad_B[i]"""
			
			for i in range(len(network)):
				network[i].W_gradient = network[i].W_gradient / len(x_train_batch)
				network[i].B_gradient = network[i].B_gradient / len(x_train_batch)
				
				network[i].W = network[i].W - lr * network[i].W_gradient
				network[i].B = network[i].B - lr * network[i].B_gradient

				network[i].W_gradient = np.zeros((network[i].output_size, network[i].input_size))
				network[i].B_gradient = np.zeros(network[i].output_size)
		
		mean_training_set_loss_current_epoch = mean_training_set_loss_current_epoch / len(x_train_set)
		training_set_loss_per_epoch.append(mean_training_set_loss_current_epoch)

		print(f"mean training loss: {mean_training_set_loss_current_epoch}")

		if x_valid_set is not None:
			for validation_index in range(len(x_valid_set)):
				x = x_valid_set[validation_index]
				y = y_valid_set[validation_index]

				y_pred = x

				for l in network:
					y_pred = l.forward(y_pred)
				
				loss = loss_function(y_pred, y)
				mean_validation_set_loss_current_epoch = mean_validation_set_loss_current_epoch + loss
			
			mean_validation_set_loss_current_epoch = mean_validation_set_loss_current_epoch / len(x_valid_set)
			validation_set_loss_per_epoch.append(mean_validation_set_loss_current_epoch)
			print(f"mean validation loss: {mean_validation_set_loss_current_epoch}")
	
	print("-- training complete --")
		
		





	