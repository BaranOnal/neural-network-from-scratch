import numpy as np
from . import utils
from . import activations
from . import losses

class NeuralNetwork:

    def __init__(self, X, Y, layers_dims, activation_function = "relu", loss_function = "squared_loss", learning_rate = 1e-3):
        self.x_train, self.x_test, self.y_train, self.y_test = utils.split_data(X, Y, test_size=0.2)

        activation_class = activations.activations[activation_function]
        self.loss_function = losses.losses[loss_function]()
        self.layers = []
        self.learning_rate = learning_rate
        for i in range(len(layers_dims) - 2):
            print(f"Layer {i + 1} | Weight Shape: ({layers_dims[i + 1]}, {layers_dims[i]}) | (Out, In)")
            self.layers.append(Dense(layers_dims[i], layers_dims[i + 1]))
            self.layers.append(activation_class())

        self.layers.append(Dense(layers_dims[-2], layers_dims[-1]))
        if loss_function == "categorical_crossentropy":
            self.layers.append(activations.Softmax())
        else:
            self.layers.append(activations.Sigmoid())


    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, y, y_hat):
        if isinstance(self.loss_function, losses.BinaryCrossEntropy):
            # Sigmoid + BCE shortcut
            dZ = y_hat - y

            # skip the last activation layer (Sigmoid)
            for layer in reversed(self.layers[:-1]):
                dZ = layer.backward(dZ)
            return dZ
        else:
            dA = self.loss_function.derivative(y, y_hat) # dA[l] from loss func
            # last layer (sigmoid + bce shortcut) bypass
            #dA = y_hat - y
            # for layer in reversed(self.layers[:-1]):  #skips the last activation layer (sigmoid)

            for layer in reversed(self.layers):
                dA = layer.backward(dA) #da -> first dZ[l] from activation func. then dense return dA[l - 1], then act. func. take dA[l- 1] -> return dZ[l - 1]

            return dA

    def update(self):
        for layer in self.layers:
            if isinstance(layer, Dense):
                layer.update(self.learning_rate)

    def create_batches(self, X, Y, batch_size):
        m = X.shape[1]
        indices = np.random.permutation(m)

        X = X[:, indices]
        Y = Y[:, indices]

        for i in range(0, m, batch_size):
            yield X[:, i:i + batch_size], Y[:, i:i + batch_size]

    def fit(self, epochs=1000, batch_size=32):
        self.history = []

        for epoch in range(epochs):
            for X_batch, Y_batch in self.create_batches(self.x_train, self.y_train, batch_size):
                y_hat = self.forward(X_batch)
                self.backward(Y_batch, y_hat)
                self.update()

            y_hat_full = self.forward(self.x_train)
            current_cost = self.loss_function.compute(self.y_train, y_hat_full)
            self.history.append(current_cost)

            if epoch % 100 == 0:
                y_hat_full = self.forward(self.x_train)
                cost = self.loss_function.compute(self.y_train, y_hat_full)
                print(f"Epoch {epoch} | Cost: {cost:.5f}")


class Layer:
    pass


class Dense(Layer):
    def __init__(self, n_inputs, n_outputs):
        self.inputs = n_inputs
        self.outputs = n_outputs
        self.weights = np.random.randn(n_outputs, n_inputs) * 0.01
        self.bias = np.zeros((n_outputs, 1))
        self.input = None
        self.output = None
        self.dw = None
        self.db = None

    def forward(self, data):

        self.input = data #A[l - 1]
        self.output = np.dot(self.weights, self.input) + self.bias
        return self.output

    def backward(self, dZ):
        m = dZ.shape[1]

        self.dw = (1/m) * np.dot(dZ, self.input.T) # self.input = A[l - 1]
        self.db = (1/m) * np.sum(dZ, axis=1, keepdims=True)

        dA = np.dot(self.weights.T, dZ)
        return dA

    def update(self, lr):
        self.weights -= lr * self.dw
        self.bias -= lr * self.db

import matplotlib.pyplot as plt

def plot_training(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history, color='blue', linewidth=2)
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    n_x = 5
    m = 1000
    X = np.random.randn(n_x, m)

    # If the sum of its first two > 0, its 1; else, its 0.
    Y = ((X[0, :] + X[1, :]) > 0).astype(float).reshape(1, m)

    layers_dims = [5, 8, 1]

    print("--- Model Running ---")
    model = NeuralNetwork(X, Y, layers_dims,
                          activation_function="relu",
                          loss_function="binary_crossentropy",
                          learning_rate=0.01)

    model.fit(epochs=1000, batch_size=32)


    print("\n--- Test Result ---")
    test_pred = model.forward(model.x_test)

    # Threshold = 0.5
    pred = (test_pred > 0.5).astype(int)
    acc = np.mean(pred == model.y_test)

    print(f"Test example: {model.x_test.shape[1]}")
    print(f"Total Cost: {model.loss_function.compute(model.y_test, test_pred):.5f}")
    print(f"Accuracy: %{acc * 100:.2f}")


    plot_training(model.history)