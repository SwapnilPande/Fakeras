
from abc import ABC, abstractmethod
import numpy as np

class Optimizer:
    @abstractmethod
    def compile(self, layers):
        """
            Compile the models and necessary parameters for training
        """
        raise NotImplementedError

    @abstractmethod
    def updateWeights(self, layers):
        """
            Compile the models and necessary parameters for training
        """
        raise NotImplementedError

class SGD(Optimizer):
    # Momentum defaults to 0.0, which is just standard SGD
    def __init__(self, lr = 0.01, momentum = 0.0):
        super().__init__()
        self.lr = lr
        self.momentum = momentum

    def compile(self, layers):
        # Initialize weight and bias velocity lists
        self.weightVelocities = [None] * len(layers)
        self.biasVelocities = [None] * len(layers)

        # Initialize velocity numpy arrays for each trainable layer
        for i, layer in enumerate(layers):
            # Check to skip layers like dropout
            if(layer.trainable):
                # Initialize all velocities to zero
                self.weightVelocities[i] = np.zeros(layers.w.shape)
                self.biasVelocities[i] = np.zeros(layers.b.shape)

    def updateWeights(self, layers):
        # Iterate over all layers
        for weightVelocity, biasVelocity, layer in zip(self.weightVelocities, self.biasVelocities, layers):
            # Perform weight update if layer is trainable
            if(layer.trainable):
                weightVelocity *= self.momentum
                weightVelocity += (1-self.momentum)*layer.dW

                biasVelocity *= self.momentum
                biasVelocity += (1-self.momentum)*layer.dB

                layer.w -= self.lr*weightVelocity
                layer.b -= self.lr*biasVelocity