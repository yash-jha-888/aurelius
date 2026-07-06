# Aurelius

Aurelius is a deep learning framework written in modern C++ from first principles.

I started this project to understand what happens underneath libraries like PyTorch instead of treating them as black boxes. Every major building block is implemented manually, from the mathematics to the engineering, with the long-term goal of growing Aurelius into a complete deep learning framework.

One of the principles behind Aurelius is simple: every abstraction I use should be one I've implemented myself first.

This release focuses on the fundamentals: dense neural networks, backpropagation, optimization, data pipelines, and training infrastructure. Future releases will add automatic differentiation, a tensor library, CNNs, GPU acceleration, and eventually a GPT-style transformer built entirely from scratch.


---

## Current Features

### Neural Networks
-  Configurable fully-connected (dense) architectures
-  Manual forward propagation
-  Manual backpropagation
-  ReLU activation
-  Softmax activation
-  Cross-entropy loss
-  Combined Softmax + Cross-Entropy gradient

### Training
-  Mini-batch SGD
-  Accuracy metric
-  Early stopping
-  Best model checkpoint restoration

### Data Pipeline
-  MNIST binary dataset loader
-  Dataset shuffling
-  Train / Validation / Test splitting

### Engineering
-  Modular architecture
-  Modern C++17
-  Eigen-based tensor operations
-  CMake build system

---

## Current Results

Training on the MNIST handwritten digit dataset:

- Architecture: **784 → 128 → 64 → 10**
- Optimizer: **Mini-batch SGD**
- Loss: **Cross Entropy**
- Test Accuracy: **94.45%**

---

## Project Structure

```
Aurelius/
│
├── include/
│   ├── Activation.h
│   ├── Dataset.h
│   ├── DenseLayer.h
│   ├── Loss.h
│   └── Model.h
│
├── src/
│   ├── Activation.cpp
│   ├── Dataset.cpp
│   ├── DenseLayer.cpp
│   ├── Loss.cpp
│   ├── Model.cpp
│   └── main.cpp
│
├── data/
├── CMakeLists.txt
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
- [ ] Adam optimizer
- [ ] Learning rate scheduling
- [ ] Xavier / He initialization
- [ ] Dropout
- [ ] L2 regularization

### v0.4
- [ ] Convolutional Neural Networks
- [ ] Batch Normalization
- [ ] Model serialization
- [ ] More datasets

### v1.0
- [ ] Automatic differentiation engine
- [ ] Tensor abstraction
- [ ] GPU acceleration
- [ ] Transformer implementation
- [ ] GPT trained completely from scratch