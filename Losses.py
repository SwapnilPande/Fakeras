import numpy as np

class BinaryCrossEntropy:
    def loss(y, a):
        return -1*y * np.log(a) - (1-y) * np.log(1-a)

    def gradient(y, a):
        return (-1*y/a + (1-y)/(1-a))
