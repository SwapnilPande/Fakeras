import numpy as np
import time
import Fakeras.Regularizers

#Base class for layers
class Layer:
    def __init__(self, neurons, activation, regularizer = None):
        self.neurons = neurons
        self.w  = None
        self.b = None
        self.activation = activation
        self.nextLayer = None
        self.prevLayer = None
        self.regularizer = regularizer

    def __len__(self):
        return self.neurons

class Dense(Layer):
    def __init__(self, neurons, activation, regularizer = None):
        super().__init__(neurons, activation, regularizer)

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
        self.dW = np.dot(dZ, self.prevA.transpose())
        # Regularization term if layer has regularizer
        if(self.regularizer is not None):
            self.dW += self.regularizer.gradient(self.w)
        # Normalize weight gradient to input size
        self.dW *= 1/self.m

        self.dB = 1/self.m * np.sum(dZ)

        # Pass dA to previous layer unless previous layer is Input
        if(type(self.prevLayer) is not Input):
            dA = np.dot(self.w.transpose(), dZ)
            self.prevLayer.backProp(dA)

    def updateWeights(self, lr):
        self.w -= lr*self.dW
        self.b -= lr*self.dB


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

class Droupout(Layer):
    def __init__(self, keepProbability):
        # Init to no neurons with no activation temporarily
        # Number neurons will be defined in the compile
        super().__init__(None, None)

        # Define keep probability
        self.keepProbability = keepProbability


    def compile(self, prevLayer, nextLayer):
        self.prevLayer = prevLayer
        self.neurons = len(prevLayer)
        self.nextLayer = nextLayer

    def forwardProp(self, prevA):
        # Create dropout mask filled with random values in range [0,1)
        self.mask = np.random.rand(prevA.shape)

        # Set all values < (1 - keepProbability) to 0
        # These are the neurons that will be dropped
        self.mask[self.mask <= (1-self.keepProbability)] = 0

        # Set all other values to 1, these are values to keep
        self.mask[self.mask > 0] = 1

        # Apply mask and pass to next layer
        if(self.nextLayer != None):
            return self.nextLayer.forwardProp(self.mask * prevA)

        # If there is not another layer, return tensor after applying activation function
        return self.activation.activation(self.mask * prevA)

    def backProp(self, prevdA):
        # Apply mask and pass to previous layer if previous layer is not input
        if(type(self.prevLayer) is not Input):
            dA = np.dot(self.w.transpose(), dZ)
            self.prevLayer.backProp(self.mask*prevdA)

    # Function does nothing, defined for compatibility with Model class
    def updateWeights(self, lr):
        return
