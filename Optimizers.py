
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
                self.weightVelocities[i] = np.zeros(layer.w.shape)
                self.biasVelocities[i] = np.zeros(layer.b.shape)

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

class Adam(Optimizer):
    # Momentum defaults to 0.0, which is just standard SGD
    def __init__(self, lr = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 10**-8):
        super().__init__()
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0

    def compile(self, layers):
        # Initialize weight and bias velocity lists
        self.weightS = [None] * len(layers)
        self.biasS = [None] * len(layers)

        self.weightR = [None] * len(layers)
        self.biasR = [None] * len(layers)

        # Initialize velocity numpy arrays for each trainable layer
        for i, layer in enumerate(layers):
            # Check to skip layers like dropout
            if(layer.trainable):
                # Initialize all velocities to zero
                self.weightS[i] = np.zeros(layers.w.shape)
                self.biasS[i] = np.zeros(layers.b.shape)
                self.weightR[i] = np.zeros(layers.w.shape)
                self.biasR[i] = np.zeros(layers.b.shape)

    def updateWeights(self, layers):
        # Iterate over all layers
        for weightS, biasS, weightR, biasR, layer in zip(self.weightS, self.biasS, self.weightR, self.biasR, layers):
            # Perform weight update if layer is trainable
            if(layer.trainable):
                weightS = self.beta1 * weightS + (1 - self.beta1) * layer.dW
                biasS = self.beta1 * biasS + (1 - self.beta1) * layer.dB

                weightR = self.beta2 * weightR + (1 - self.beta2) * np.square(layer.dW)
                biasR = self.beta2 * biasR + (1 - self.beta2) * np.square(layer.dB)

                # Bias Correction
                weightSHat = weightS/(1-self.beta1**self.t)
                biasSHat = biasS/(1-self.beta1**self.t)
                weightRHat = weightR/(1-self.beta2**self.t)
                biasRHat = biasR/(1-self.beta2**self.t)

                layer.w -= self.lr * weightSHat / (np.sqrt(weightRHat) + self.epsilon)
                layer.b -= self.lr * biasSHat / (np.sqrt(biasRHat) + self.epsilon)

        self.t += 1