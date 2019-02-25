import numpy as np
import time

#Base class for layers
class Layer:
    def __init__(self, neurons, activation):
        self.neurons = neurons
        self.w  = None
        self.b = None
        self.activation = activation
        self.nextLayer = None
        self.prevLayer = None

class Dense(Layer):
    def __init__(self, neurons, activation):
        super().__init__(neurons, activation)

    def compile(self, prevLayer, nextLayer):
        self.prevLayer = prevLayer
        self.nextLayer = nextLayer

        # Initialize weights to value between 0-0.01
        # Dimension of w: N[l] x N[l-1]
        self.w = np.random.rand(self.neurons, len(self.prevLayer))/100

        # Dimension of w: N[l] x 1
        self.b = np.random.rand(self.neurons, 1)/100

    def forwardProp(self, prevA):
        # Save activation from previous layer to calculate gradient
        self.prevA = prevA

        # Store number of training examples for backprop
        self.m = self.prevA.shape[1]

        # Apply linear weights and biases
        self.Z = np.dot(self.w, self.prevA) + self.b

        # Apply activation function and pass tensor to next layer
        if(self.nextLayer != None):
            return self.nextLayer.forwardProp(self.activation.activation(self.Z))

        # If there is not another layer, return tensor after applying activation function
        return self.activation.activation(self.Z)

    def backProp(self, prevdA):
        # Calculate dZ
        dZ = prevdA * self.activation.gradient(self.Z)

        # Calculate gradients with respect to weights and biases
        self.dW = 1/self.m * np.dot(dZ, self.prevA.transpose())
        self.dB = 1/self.m * np.sum(dZ)

        # Return derivative with respect to activation of previous layer
        dA = np.dot(self.w.transpose(), dZ)
        self.prevLayer.backProp(dA)

    def updateWeights(self, lr):
        self.w = self.w - lr*self.dW
        self.b = self.b - lr*self.dB

    def __len__(self):
        return self.neurons

class Input(Layer):
    def __init__(self, inputDim):
        super().__init__(inputDim, None)

    def compile(self, prevLayer, nextLayer):
        self.prevLayer = prevLayer
        self.nextLayer = nextLayer

    def forwardProp(self, a):
        return self.nextLayer.forwardProp(a)

    def backProp(self, prevdA):
        return

    def __len__(self):
        return self.neurons
