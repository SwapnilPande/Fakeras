import numpy as np


# L2 Regularization class
class L2:
    def __init__(self, alpha):
        self.alpha = alpha

    # Return regularization penalty
    def regularizer(self, w):
        return self.alpha/2*np.sum(w*w)

    def gradient(self, w):
        return self.alpha * w
