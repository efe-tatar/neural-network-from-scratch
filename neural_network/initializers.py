import numpy as np

def xavier_initializer(input_size, output_size):
    l = np.sqrt(6 / (input_size + output_size))
    W = np.random.uniform(-l, l, (output_size, input_size))
    # B = np.random.uniform(-l, l, output_size)
    # B = np.zeros(output_size)
    B = np.random.uniform(-0.1, 0.1, output_size)
    return W, B