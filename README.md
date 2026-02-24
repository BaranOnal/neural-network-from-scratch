# Neural Network From Scratch (NumPy) 

A minimal neural network framework built with NumPy to understand deep learning from first principles.

This project rebuilds core deep learning components from the ground up to better understand how modern frameworks work internally.

---

## Features

- **Dense (Fully Connected) Layers**
- **ReLU, Sigmoid, Softmax activations**
- **MSE, Binary Cross Entropy, Categorical Cross Entropy losses**
- **Mini-batch Gradient Descent**
- **He and Xavier initialization**
- **L2 regularization**
- **SGD, Momentum, RMSprop, Adam optimizers**

---

## Results

### XOR Classification
- Achieved **100% accuracy**
- Successfully learned non-linear decision boundary

![XOR Loss Curve](images/xor_loss.png)

---

### MNIST Digit Classification
- Achieved approximately **~95% test accuracy**

![MNIST Predictions](images/mnist_predictions.png)

> Top: Model predictions  
> Bottom: Training loss curve  

---
## Project Structure

```
neural-network-from-scratch/
│
├── nn/                 
│   ├── layers.py
│   ├── activations.py  
│   ├── losses.py  
│   ├── optimizers.py     
│   └── utils.py        
│
├── examples/           
│   ├── train_xor.py
│   └── train_mnist.py
│
├── images/
│   ├── xor_loss.png
│   └── mnist_predictions.png
│
└── README.md
```

---

##  Future Improvements
- Dropout  
- Early Stopping

---

## Purpose

This project is not intended to replace PyTorch or TensorFlow.

It exists to build intuition.

When you understand this code, deep learning frameworks stop feeling like black boxes.