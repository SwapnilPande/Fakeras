import Fakeras.Layers
import Fakeras.Losses
import numpy as np
import time
import tqdm
import math

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
        return self.loss.loss(y_train, self.a, self.layers)

    # Run backpropagation to calculate gradients
    def __backProp__(self, y_train):
        # Calculate gradients for all layers recursively
        self.layers[-1].backProp(self.loss.gradient(y_train, self.a))

    # Update weights of each layer
    def __updateWeights__(self):
        for i in range(1, len(self.layers)):
            self.layers[i].updateWeights(self.lr)

    # Train the model using the gradient descent algorithm
    def fit(self, x_train, y_train, batch_size = None, epochs = 1, x_val = None, y_val = None, valFreq = 5):
        # If no batch size specified, set batch size to the number of data points to run GD
        if(batch_size is None):
            batch_size = x_train.shape[1]

        # Calculate the number of minibatches to be generated
        numSteps = math.ceil(x_train.shape[1]/batch_size)

        # List containing loss at each iteration of SGD
        lossPerIteration = [None] * epochs

        # Enable Dropout Layers
        for layer in self.layers:
            if type(layer) is Fakeras.Layers.Dropout:
                layer.enable()


        # Iterate number of epochs
        for i in range(epochs):
            # Shuffle data to split into batches
            # Create a random permutation of number of examples
            randOrder = np.random.permutation(x_train.shape[1])

            # Rearrange the data according to the random index
            x_train_shuffled = x_train[:, randOrder]
            y_train_shuffled = y_train[:, randOrder]

            pbar = tqdm.trange(numSteps, desc = "Epoch {i}/{epochs}".format(i = i, epochs = epochs))
            for j in pbar:
                # Index to retrieve X
                startIndex = j*batch_size

                # Generate minibatches
                try: #Full size batch
                    minibatch_x = x_train_shuffled[:, startIndex : startIndex + batch_size]
                    minibatch_y = y_train_shuffled[:, startIndex : startIndex + batch_size]
                except IndexError: #Retrieve small batch at the end of the array
                    minibatch_x = x_train_shuffled[:, startIndex:]
                    minibatch_y = y_train_shuffled[:, startIndex:]

                self.__forwardProp__(minibatch_x)

                # Store loss
                #lossPerIteration[j + numSteps*i] = self.__calculateLoss__(minibatch_y)
                loss = self.__calculateLoss__(minibatch_y)


                self.__backProp__(minibatch_y)

                self.__updateWeights__()


                # Display loss on progress bar, display validation loss if reached val frequency is correct
                if(j == numSteps -1 and i % valFreq == 0 and x_val is not None and y_val is not None):
                    valLoss,_ = self.evaluate(x_val, y_val)
                    pbar.set_postfix(loss = loss, val_loss = valLoss)
                else:
                    pbar.set_postfix(loss = loss)

            lossPerIteration[i] = loss





        # Print final training loss
        print("Final Training Loss: {finalTrainingLoss}".format(finalTrainingLoss = lossPerIteration[-1]))

        return lossPerIteration

    def evaluate(self, x, y):
        # Disable dropout layers
        for layer in self.layers:
            if type(layer) is Fakeras.Layers.Dropout:
                layer.disable()

        # Run forward prop to generate predictions
        self.__forwardProp__(x)

        loss = self.__calculateLoss__(y)

        predictions = np.copy(self.a)
        predictions[predictions > 0.5] = 1
        predictions[predictions <= 0.5] = 0

        error = 100*np.count_nonzero(predictions - y)/x.shape[1]

        return loss, error


    def predict(self, x):
        # Disable dropout layers
        for layer in self.layers:
            if type(layer) is Fakeras.Layers.Dropout:
                layer.disable()

        self.__forwardProp__(x)

        predictions = np.copy(self.a)
        predictions[predictions > 0.5] = 1
        predictions[predictions <= 0.5] = 0

        return predictions


