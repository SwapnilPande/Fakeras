
import Fakeras.Layers
import Fakeras.Losses
import numpy as np

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
        for i in range(1, len(self.layers)):
            self.layers[i].initWeights(len(self.layers[i-1]))

    # Run forward propagation algorithm
    def __forwardProp__(self, x):
        # a stores the numpy array propagating through nn
        self.a = x

        # Propagate through each hidden layer and output layer
        for i in range(1, len(self.layers)):
            self.a = self.layers[i].forwardProp(self.a)

    # Calculate the loss
    def __calculateLoss__(self, y_train):
        return self.loss.loss(y_train, self.a)

    # Run backpropagation to calculate gradients
    def __backProp__(self, y_train):
        # Calculate dA for loss function
        grad = self.loss.gradient(y_train, self.a)

        # Calculate gradients for each layer
        for i in reversed(range(1, len(self.layers))):
            grad = self.layers[i].backProp(grad)

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


