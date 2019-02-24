import numpy as np

class BinaryCrossEntropy:
    def loss(self, y, a):
        return np.sum(-1*y * np.log(a) - (1-y) * np.log(1-a))/a.shape[1]

    def gradient(self, y, a):
        return (-1*y/a + (1-y)/(1-a))
