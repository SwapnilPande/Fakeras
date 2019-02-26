import numpy as np
import Fakeras.Layers

class BinaryCrossEntropy:
    def loss(self, y, a, layers):
        # Loss without regularization
        j = np.sum(-1*y * np.log(a) - (1-y) * np.log(1-a))/a.shape[1]

        # Regularization term
        reg = 0
        # Calculate loss penalty for each layer
        for i in range(1, len(layers)):
            if(layers[i].regularizer is not None):
                reg += layers[i].regularizer.regularizer(layers[i].w)

        return j + reg/a.shape[1]

    def gradient(self, y, a):
        return (-1*y/a + (1-y)/(1-a))
