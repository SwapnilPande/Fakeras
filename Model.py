
import Layers
import Losses

class Model:

    def __init__(self):
        # List containing layers for the neural network
        self.layers = []

    def __forwardProp__(self, x_train):
        # Input layer
        a = self.layers[0].forwardProp(x_train)
        # Propagate through each hidden layer and output layer
        for i in range(1, len(layers)):
            a = self.layers[i].forwardProp(a)


    def __backProp__(self):
        grad =

    def __calculateLoss__(self):

    def __updateWeights__(self):

    def fit(self, x_train, y_train, epochs):
        for i in range(epochs):
            self.__forwardProp__()

