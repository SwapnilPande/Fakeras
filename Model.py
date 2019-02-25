
import Fakeras.Layers
import Fakeras.Losses
import numpy as np
import time

# Model class defines the entire model structure and has functions for fitting and generating predictions
# To add layers, call Model.add() and pass a Layer object
# Once structure is defined, Model.compile() will initialize weights
# Call Model.fit() to train the model
class Model:

    # Initialize empty list to store layers
    def __init__(self):
        # List containing layers for the neural network
        self.layers = []


        # Init variable to store numpy array to propagate through network
        self.a = None

    # Append a layer to the end of the layer list
    def add(self, layer):
        self.layers.append(layer)

    def compile(self, loss, lr):
        self.loss = loss
        self.lr = lr

        # Compile input layer separately
        # There is no previous layer
        self.layers[0].compile(None, self.layers[1])

        # Compile each hidden layer
        for i in range(1, len(self.layers)-1):
            self.layers[i].compile(self.layers[i-1], self.layers[i+1])

        # Compile output layer separately
        # There is no next layer
        self.layers[-1].compile(self.layers[-2], None)

    # Run forward propagation algorithm
    def __forwardProp__(self, x):
        # a stores the numpy array propagating through neural network
        self.a = self.layers[0].forwardProp(x)

    # Calculate the loss
    def __calculateLoss__(self, y_train):
        return self.loss.loss(y_train, self.a)

    # Run backpropagation to calculate gradients
    def __backProp__(self, y_train):
        # Calculate gradients for all layers recursively
        self.layers[-1].backProp(self.loss.gradient(y_train, self.a))

    # Update weights of each layer
    def __updateWeights__(self):
        for i in range(1, len(self.layers)):
            self.layers[i].updateWeights(self.lr)

    # Train the model using the gradient descent algorithm
    def fit(self, x_train, y_train, epochs):
        # List containing loss at each iteration of GD
        lossPerIteration = [None] * epochs

        # Iterate number of epochs
        for i in range(epochs):
            self.__forwardProp__(x_train)

            lossPerIteration[i] = (self.__calculateLoss__(y_train))

            self.__backProp__(y_train)

            self.__updateWeights__()

            if(i % 100 == 0):
                print("Epoch: {epoch}\tTraining Loss: {train_loss}".format(epoch=i, train_loss=lossPerIteration[i]))

        return lossPerIteration

    def predict(self, x):
        self.__forwardProp__(x)
        return self.a


