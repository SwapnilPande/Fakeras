import numpy as np

class Dense:
    def __init__(self, neurons, activation):
        self.neurons = neurons
        self.w  = None
        self.b = None
        self.activation = activation

    def initWeights(self, prevLayerDim):
        # Initialize weights to value between 0-0.01
        # Dimension of w: N[l] x N[l-1]
        self.w = np.random.rand(self.neurons, prevLayerDim)/100

        # Dimension of w: N[l] x 1
        self.b = np.random.rand(self.neurons, 1)/100

    def forwardProp(self, prevA):
        # Save activation from previous layer to calculate gradient
        self.prevA = prevA

        # Store number of training examples for backprop
        self.m = self.prevA.shape[1]

        # Apply linear weights and biases
        self.Z = np.dot(self.w, self.prevA) + self.b

        # Return tensor after activation is applied
        return self.activation.activation(self.Z)

    def backProp(self, prevdA):
        # Calculate dZ
        dZ = prevdA * self.activation.gradient(self.Z)

        # Calculate gradients with respect to weights and biases
        self.dW = 1/self.m * np.dot(dZ, self.prevA.transpose())
        self.dB = 1/self.m * np.sum(dZ)

        # Return derivation with respect to activation of previous layer
        return np.dot(self.w.transpose(), dZ)

    def updateWeights(self, lr):
        self.w = self.w - lr*self.dW
        self.b = self.b - lr*self.dB

    def __len__(self):
        return self.neurons

class Input:
    def __init__(self, inputDim):
        self.inputDim = inputDim

    def __len__(self):
        return self.inputDim
