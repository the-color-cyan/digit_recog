import numpy as np


class Network(object):

    def __init__(self, sizes):  # list sizes = # of neurons in each layer
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Returns the network's output if a is the input"""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a


# Helper functions
# sigmoid function
def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))
