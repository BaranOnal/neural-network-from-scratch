from nn.layers import *

X_xor = np.array([[0, 0, 1, 1],
                  [0, 1, 0, 1]])

Y_xor = np.array([[0, 1, 1, 0]])

layers_dims = [2, 4, 1]

print("--- Model XOR ---")

model = NeuralNetwork(X_xor, Y_xor, layers_dims,
                      activation_function="relu",
                      loss_function="binary_crossentropy",
                      learning_rate=0.5)

model.fit(epochs=2000)

print("\n--- XOR Training Results ---")

predictions = model.forward(X_xor)

print("Input:\n", X_xor)
print("True Values:", Y_xor)
print("Predict (Raw):", np.round(predictions, 4))
print("Predict (Binary):", (predictions > 0.5).astype(int))

accuracy = np.mean((predictions > 0.5) == Y_xor)
print(f"\nFinal Accuracy: %{accuracy * 100}")

plot_training(model.history)