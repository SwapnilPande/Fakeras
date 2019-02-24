class Dense:
    def __init__(self, neurons, activation):
        self.w  = None
        self.b = None
        self.activation = activation

    def forwardProp(prevA):
        # Save activation from previous layer to calculate gradient
        self.prevA = prevA

        # Store number of training examples for backprop
        self.m = prevA.shape(1)

        # Apply linear weights and biases
        self.Z = np.dot(w, prevA) + b

        # Return tensor after activation is applied
        return self.activation.activation(self.Z)

    def backProp(prevdA):
        # Calculate dZ
        dZ = prevdA * self.activation.gradient(self.Z)

        # Calculate gradients with respect to weights and biases
        self.dW = 1/self.m * np.dot(dZ, self.prevA.transpose())
        self.dB = 1/self.m * np.sum(dZ)

        # Return derivation with respect to activation of previous layer
        return np.dot(self.w.transpose(), dZ)