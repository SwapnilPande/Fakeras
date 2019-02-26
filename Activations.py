import numpy as np

class Relu:
    def activation(self, input):
        input[input <= 0] = 0
        return input

    def gradient(self, input):
        input[input <= 0] = 0
        input[input > 0] = 1
        return input

class Tanh:
    def activation(self, input):
        self.a = np.tanh(input)
        return self.a

    def gradient(self, input):
        return 1 - self.a*self.a

class Sigmoid:
    def activation(self, input):
        self.a = 1/(1+np.exp(-1*input))
        return self.a

    def gradient(self, input):
        return self.a*(1-self.a)
