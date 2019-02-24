import numpy as np

class Relu:
    def applyActivation(self, input):
        input[input <= 0] = 0
        return input

    def applyDerivative(self, input):
        input[input <= 0] = 0
        input[input > 0] = 1
        return input

class Sigmoid:
    def applyActivation(input):
        return np.tanh(input)

    def applyDerivative(input):
        a = tanh(input)
        return 1 - a*a