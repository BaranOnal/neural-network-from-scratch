import numpy as np
from nn.layers import NeuralNetwork, plot_training


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
                          learning_rate=0.001,
                          optimizer="Adam",
                          initialization="he"
                          )

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
