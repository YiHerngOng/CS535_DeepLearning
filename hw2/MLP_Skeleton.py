"""
Yi Herng Ong
"""


from __future__ import division
from __future__ import print_function
import math
import sys
try:
   import _pickle as pickle
except:
   import pickle

import numpy as np
import sympy  # added to solve partial derivative

# This is a class for a LinearTransform layer which takes an input 
# weight matrix W and computes W x as the forward step
class LinearTransform(object):

    def __init__(self, W, b):
	# DEFINE __init function
        self.w = w
        self.b = b

    def forward(self, x):
	# DEFINE forward function
        return (np.matmul(self.w, x) + self.b)

    def backward(
        self, 
        grad_output, 
        learning_rate=0.0, 
        momentum=0.0, 
        l2_penalty=0.0,
    ):
	# DEFINE backward function
        return np.dot(grad_output, self.w) # not sure it is dot product


# ADD other operations in LinearTransform if needed
    


# This is a class for a ReLU layer max(x,0)
class ReLU(object):

    def forward(self, x):
	# DEFINE forward function
        self.x = x
        return max(self.x, 0)

    def backward(
        self, 
        grad_output, 
        learning_rate=0.0, 
        momentum=0.0, 
        l2_penalty=0.0,
    ):
    # DEFINE backward function
        if self.x > 0:
            return grad_output * 1
        if self.x = 0:
            return grad_output * [0, 1] # not right
        else:
            return 0
# ADD other operations in ReLU if needed

# This is a class for a sigmoid layer followed by a cross entropy layer, the reason 
# this is put into a single layer is because it has a simple gradient form
class SigmoidCrossEntropy(object):
	def forward(self, x):
		# DEFINE forward function
        return (1 / (1 + math.exp(-x)))

	def backward(
	    self, 
	    grad_output, 
		learning_rate=0.0,
		momentum=0.0,
		l2_penalty=0.0
	):
		# DEFINE backward function
        grad_network = math.exp(-x) / (1+math.exp(-x))**2
        return grad_output * grad_network

# ADD other operations and data entries in SigmoidCrossEntropy if needed


# This is a class for the Multilayer perceptron
class MLP(object):

    def __init__(self, input_dims, hidden_units):
    # INSERT CODE for initializing the network
        self.input_dims = input_dims
        self.hidden_units = hidden_units

    def train(
        self, 
        x_batch, 
        y_batch, 
        learning_rate, 
        momentum,
        l2_penalty,
    ):
	# INSERT CODE for training the network

    def evaluate(self, x, y):
	# INSERT CODE for testing the network

# ADD other operations and data entries in MLP if needed
    def loss_function(self, y, z):
        '''Implement with cross entropy loss function'''
        cross_entropy_loss = y*math.log(z) + (1 - y)*math.log(1-z)
        return cross_entropy_loss

if __name__ == '__main__':
    if sys.version_info[0] < 3:
        data = pickle.load(open('cifar_2class_py2.p', 'rb'))
	else:
	    data = pickle.load(open('cifar_2class_py2.p', 'rb'), encoding='bytes')

    train_x = data['train_data']
    train_y = data['train_labels']
    test_x = data['test_data']
    test_y = data['test_labels']
	
    num_examples, input_dims = train_x.shape
	# INSERT YOUR CODE HERE
	# YOU CAN CHANGE num_epochs AND num_batches TO YOUR DESIRED VALUES
	num_epochs = 10
	num_batches = 1000
    mlp = MLP(input_dims, hidden_units)

    for epoch in xrange(num_epochs):

	# INSERT YOUR CODE FOR EACH EPOCH HERE

        for b in xrange(num_batches):
			total_loss = 0.0
			# INSERT YOUR CODE FOR EACH MINI_BATCH HERE
			# MAKE SURE TO UPDATE total_loss
            print(
                '\r[Epoch {}, mb {}]    Avg.Loss = {:.3f}'.format(
                    epoch + 1,
                    b + 1,
                    total_loss,
                ),
                end='',
            )
            sys.stdout.flush()
		# INSERT YOUR CODE AFTER ALL MINI_BATCHES HERE
		# MAKE SURE TO COMPUTE train_loss, train_accuracy, test_loss, test_accuracy
        print()
        print('    Train Loss: {:.3f}    Train Acc.: {:.2f}%'.format(
            train_loss,
            100. * train_accuracy,
        ))
        print('    Test Loss:  {:.3f}    Test Acc.:  {:.2f}%'.format(
            test_loss,
            100. * test_accuracy,
        ))
