import numpy as np

class DenseLayer:
	def __init__(self, input_size, output_size, act_func, derv_act_func):
		self.input_size = input_size
		self.output_size = output_size
		self.act_func = act_func
		self.derv_act_func = derv_act_func

		self.W = np.random.uniform(-1, 1, (output_size, input_size)) * (1 / np.sqrt(input_size))
		self.B = np.random.uniform(-0.1, 0.1, output_size)

		self.W_gradient = np.zeros((output_size, input_size))
		self.B_gradient = np.zeros(output_size)
	
	def forward(self, x:np.ndarray):
		self.z = self.W @ x + self.B
		self.a = self.act_func(self.z)
		return self.a