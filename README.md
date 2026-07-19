# Aurelius

Aurelius is a deep learning framework written in modern C++ from first principles.

I started this project to understand what happens underneath libraries like PyTorch instead of treating them as black boxes. Every major building block is implemented manually, from the mathematics to the engineering, with the long-term goal of growing Aurelius into a complete deep learning framework.

One of the principles behind Aurelius is simple: every abstraction I use should be one I've implemented myself first.

The project implements the core components of neural network training without relying on existing machine learning frameworks. Its primary goal is to provide a reference implementation of how modern deep learning systems work internally while maintaining a modular, extensible architecture.

The framework currently supports fully connected neural networks trained on the MNIST handwritten digit dataset. Each major subsystem, including activations, weight initialization, loss functions, logging, and optimization, is designed as an independent abstraction that can be extended without modifying the rest of the framework.


---

## Features

### Neural networks

- Configurable fully connected architectures
- Forward propagation
- Backpropagation
- ReLU activation
- Leaky ReLU activation
- Softmax output layer

### Training

- Mini-batch stochastic gradient descent (SGD)
- Cross-entropy loss
- Early stopping
- Best-model checkpoint restoration
- Training and validation evaluation
- Official MNIST test-set evaluation

### Framework architecture

- Activation abstraction
- Initializer abstraction
- Modular layer design
- Timestamped experiment logging
- CSV training metrics

### Data

- IDX dataset parser
- Official MNIST train/test split
- Dataset shuffling
- Train/validation split

---

## Architecture

Aurelius separates each major training component behind a common interface.

- Activation functions
- Weight initializers
- Optimizers
- Loss functions
- Layers

This design allows components to be replaced independently while keeping the training pipeline unchanged.

---

## Results

Current benchmark on MNIST:

| Metric | Value |
|--------|------:|
| Architecture | 784 → 128 → 64 → 10 |
| Optimizer | SGD |
| Activation | ReLU |
| Initializer | He |
| Test accuracy | 97.47% |

---

## Project Structure

```
Aurelius/
│
├── include/
│   ├── Activation.h
│   ├── Dataset.h
│   ├── DenseLayer.h
│   ├── Initializer.h
│   ├── Logger.h
│   ├── Loss.h
│   ├── Model.h
│   └── RunConfig.h
│
├── src/
│   ├── Activation.cpp
│   ├── Dataset.cpp
│   ├── DenseLayer.cpp
│   ├── Initializer.cpp
│   ├── Logger.cpp
│   ├── Loss.cpp
│   ├── Model.cpp
│   └── main.cpp
│
├── data/
│
├── build/
│   └── logs/
│       └── <timestamp>/
│           ├── metrics.csv
│           └── config.json
│
├── .gitignore
├── CMakeLists.txt
├── LICENSE
└── README.md
```

---

## Build

```bash
mkdir build
cd build

cmake ..
cmake --build .

./Aurelius
```

---

## Roadmap

### v0.3

- Optimizer abstraction
- JSON experiment metadata
- Training metric visualization
- L2 regularization
- Dropout

### v0.4

- Automatic differentiation
- Convolutional layers
- Batch normalization
- Model serialization

### v0.5

- Tensor library
- CUDA backend
- Transformer implementation