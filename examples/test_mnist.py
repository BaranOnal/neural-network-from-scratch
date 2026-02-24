import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder
from nn.layers import NeuralNetwork, plot_training
import matplotlib.pyplot as plt

print("MNIST is being downloaded")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)# flatten (70000, 28, 28) -> (70000,28*28)
X_raw = mnist.data.T # (784, 70000)
Y_raw = mnist.target.astype(int).reshape(-1, 1)

# Normalization
X = X_raw / 255.0

# one_hot encoding
encoder = OneHotEncoder(sparse_output=False)
Y = encoder.fit_transform(Y_raw).T # (10, 70000)

print(f"Data ready: X {X.shape}, Y {Y.shape}")


layers_dims = [784, 128, 64, 10]

model = NeuralNetwork(X, Y, layers_dims,
                      activation_function="relu",
                      loss_function="categorical_crossentropy",
                      learning_rate=0.01,
                      initialization="he",
                      optimizer="Adam"
                      )

model.fit(epochs=30, batch_size=64)

test_pred = model.forward(model.x_test)
pred_classes = np.argmax(test_pred, axis=0)
actual_classes = np.argmax(model.y_test, axis=0)

accuracy = np.mean(pred_classes == actual_classes)
print(f"\nMNIST Test Accuracy: %{accuracy * 100:.2f}")


def show_multiple_predictions(model, start_index=0):
    plt.figure(figsize=(15, 8))
    for i in range(8):
        index = start_index + i
        # 2x4 , i'th
        plt.subplot(2, 4, i + 1)

        img = model.x_test[:, index].reshape(28, 28)
        plt.imshow(img, cmap='gray')

        # [:, index:index+1] -> slide for (784, 1) shape
        output = model.forward(model.x_test[:, index:index + 1])
        pred = np.argmax(output)
        actual = np.argmax(model.y_test[:, index:index + 1])

        color = 'green' if pred == actual else 'red'
        plt.title(f"Tahmin: {pred}\nGerçek: {actual}", color=color)

    plt.tight_layout()
    plt.show()


show_multiple_predictions(model, 2000)
plot_training(model.history)