
# Assignment 3 – Multilayer Perceptrons on MNIST (Part A)

## Overview

This notebook trains a sequence of multilayer perceptron (MLP) models on the MNIST handwritten digits dataset. The focus is to:

* Compare different weight initializers (zeros, ones, random normal).
* Explore depth (2, 3, 10, 20 hidden layers).
* Study dropout placement and strength.
* Observe the effect of learning rate on convergence and overfitting.
* Use early stopping, confusion matrices and learning curves to analyze performance.

The notebook is organized into Experiments #1–11, with an additional “Experiment #11 New Trial” to reduce overfitting.

---

## Dataset

* **Source:** tf.keras.datasets.mnist
* **Input:**

  * Grayscale images of size 28 × 28.
* **Classes:** digits 0–9 (10 classes).
* **Splits used:**

  * Training: 55,000 images
  * Validation: 5,000 images
  * Test: 10,000 images
* **Preprocessing:**

  * Pixel intensities are scaled to the range [0, 1] by dividing by 255.
  * A validation set is taken from the first 5,000 training samples, the rest are used for training.

---

## What the Notebook Does

### 1. Libraries and Setup

* Imports:

  * numpy, pandas, matplotlib, time, warnings
  * tensorflow / tf.keras (Sequential, Dense, Flatten, optimizers, callbacks)
  * sklearn.metrics (confusion_matrix, classification_report)
* Sets random seeds for reproducibility:

  * np.random.seed(100) and tf.random.set_seed(100).

### 2. Loading and Scaling MNIST

* Loads the MNIST dataset using tf.keras.datasets.mnist.
* Splits the original training set into:

  * A validation set (first 5,000 samples).
  * A reduced training set (remaining samples).
* Normalizes images by dividing all pixel values by 255 so inputs lie in [0, 1].

### 3. Common Training Pipeline

Each experiment follows a similar training pattern:

1. **Reset the graph**

   * Clears any existing model/graph with tf.keras.backend.clear_session().

2. **Build the model**

   * Uses Flatten(input_shape=[28, 28]) to convert images to vectors.
   * Adds several Dense layers with ReLU activation.
   * Optionally adds Dropout layers between hidden layers.
   * Ends with Dense(10, activation='softmax') for 10-class classification.

3. **Compile**

   * Optimizer: SGD with a specified learning rate (e.g., 0.1, 0.5, 0.05).
   * Loss: SparseCategoricalCrossentropy (from logits = False).
   * Metric: accuracy.

4. **Early stopping**

   * Uses EarlyStopping on validation loss with a patience value.
   * Restores the best model weights based on validation loss.

5. **Evaluate and analyze**

   * Reports train and test loss and accuracy.
   * Computes train and test accuracy from predicted labels.
   * Prints confusion matrix and classification report for the test set.
   * Plots training vs validation accuracy and loss across epochs.

---

## Experiments with MLPs on MNIST

### Experiment #1 – 2-layer MLP with Zero Weights

* **Initializer:** all weights initialized to zero.
* **Architecture:**

  * Flatten
  * Dense(300, ReLU)
  * Dense(100, ReLU)
  * Dense(10, Softmax)
* **Optimizer:** SGD with learning rate 0.1.
* **Purpose:** show what happens when all weights start at exactly zero (symmetry problem).

### Experiment #2 – 2-layer MLP with Ones Initialization

* **Initializer:** all weights initialized to one.
* **Architecture:** same as Experiment #1 (300 → 100 → 10).
* **Optimizer:** SGD with learning rate 0.1.
* **Purpose:** see how training behaves when every weight starts from one.

### Experiment #3 – 2-layer MLP with Random Normal Initialization

* **Initializer:** random_normal.
* **Architecture:** same 2-layer MLP (300 → 100 → 10).
* **Optimizer:** SGD with learning rate 0.1.
* **Purpose:** move to a realistic initialization strategy and compare training to Experiments #1 and #2.

### Experiment #4 – Dropout on First Hidden Layer (Strong Dropout)

* **Initializer:** random_normal.
* **Architecture:**

  * Flatten
  * Dense(300, ReLU)
  * Dropout(rate = 0.5)
  * Dense(100, ReLU)
  * Dense(10, Softmax)
* **Optimizer:** SGD with learning rate 0.1.
* **Purpose:** apply strong dropout on the first hidden layer and see its effect on performance and overfitting.

### Experiment #5 – 2-layer Baseline (Random Normal, No Dropout)

* **Initializer:** random_normal.
* **Architecture:** Dense(300, ReLU) → Dense(100, ReLU) → Dense(10, Softmax).
* **Optimizer:** SGD with learning rate 0.1, up to 50 epochs with early stopping.
* **Purpose:** establish a clean baseline MLP with sensible initialization and no dropout.

### Experiment #6 – Uniform Dropout (0.1 on Both Hidden Layers)

* **Initializer:** random_normal.
* **Architecture:**

  * Dense(300, ReLU) + Dropout(0.1)
  * Dense(100, ReLU) + Dropout(0.1)
  * Dense(10, Softmax)
* **Optimizer:** SGD with learning rate 0.1.
* **Purpose:** compare light dropout on both hidden layers to the no-dropout baseline.

### Experiment #7 – Asymmetric Dropout (0.5 then 0.1)

* **Initializer:** random_normal.
* **Architecture:**

  * Dense(300, ReLU) + Dropout(0.5)
  * Dense(100, ReLU) + Dropout(0.1)
  * Dense(10, Softmax)
* **Optimizer:** SGD with learning rate 0.1.
* **Purpose:** test stronger dropout on the first hidden layer and weaker dropout on the second.

### Experiment #8 – High Learning Rate with Dropout

* **Initializer:** random_normal.
* **Architecture:** same as Experiment #7.
* **Optimizer:** SGD with learning rate 0.5 (much larger).
* **Purpose:** investigate how a larger learning rate interacts with dropout and whether it destabilizes training.

### Experiment #9 – 10 Hidden Layers (100 Units Each)

* **Initializer:** random_normal.
* **Architecture:**

  * Flatten
  * 10 × [Dense(100, ReLU)]
  * Dense(10, Softmax)
* **Optimizer:** SGD with learning rate 0.1.
* **Purpose:** explore the effect of a deeper MLP (10 hidden layers) on optimization and generalization.

### Experiment #10 – 20 Hidden Layers (100 Units Each)

* **Initializer:** random_normal.
* **Architecture:**

  * Flatten
  * 20 × [Dense(100, ReLU)]
  * Dense(10, Softmax)
* **Optimizer:** SGD with learning rate 0.1.
* **Purpose:** push depth further and compare training and validation behaviour to the 10-layer model.

### Experiment #11 – 4 Hidden Layers with Dropout 0.1

* **Initializer:** random_normal.
* **Architecture:**

  * Flatten
  * Dense(100, ReLU) + Dropout(0.1)
  * Dense(300, ReLU) + Dropout(0.1)
  * Dense(300, ReLU) + Dropout(0.1)
  * Dense(100, ReLU) + Dropout(0.1)
  * Dense(10, Softmax)
* **Optimizer:** SGD with learning rate 0.1.
* **Purpose:** use a moderately deep, symmetric MLP with light dropout to see if it generalizes well.

### Experiment #11 – New Trial (Solving Overfitting)

* **Architecture:** same as Experiment #11 above.
* **Changes:**

  * Increase dropout to 0.15.
  * Reduce learning rate to 0.05.
* **Purpose:** directly reduce overfitting observed in the previous run by increasing regularization and lowering the learning rate.
* **Evaluation:** same metrics as before (train/test accuracy, confusion matrix, classification report, learning curves).

---

## Libraries Used

* Python 3.x
* numpy, pandas, matplotlib
* tensorflow / tf.keras

  * Sequential, Dense, Flatten, Dropout
  * SGD optimizer
  * EarlyStopping callback
* scikit-learn

  * confusion_matrix
  * classification_report

---

## How to Run

1. Open the notebook in Google Colab or a local Jupyter environment.
2. Ensure the following packages are installed:

   * tensorflow
   * scikit-learn
   * pandas
   * numpy
   * matplotlib
3. Run all cells in order:

   * MNIST will be downloaded automatically from tf.keras.datasets.mnist.
4. Examine the printed metrics and plots for each experiment.

---

## Key Takeaways

* Weight initialization (zeros vs ones vs random) strongly affects whether the network can learn.
* Shallow to moderately deep MLPs (2–4 hidden layers) perform very well on MNIST.
* Very deep MLPs (10–20 layers) are harder to train with plain SGD and ReLU without additional tricks.
* Dropout and learning rate tuning are key to controlling overfitting and improving generalization.
* Early stopping based on validation loss helps automatically choose a good stopping point.

---

-----------------------------------------------------------------------------------

# Assignment 3 – Multilayer Perceptrons on CIFAR-10 (Part B)

## Overview

This notebook applies multilayer perceptron (MLP) models to the CIFAR-10 color image dataset. It extends Assignment 3 Part A from MNIST to a more challenging, high-dimensional problem.

The goals are to:

* Build fully-connected networks for 10-class image classification.
* Compare different architectures (width/depth).
* Explore dropout and learning rate choices.
* Compare SGD vs Adam optimizers.
* Evaluate performance using accuracy, confusion matrices, and learning curves.

All three models are grouped under Experiment #12, as three different trials on CIFAR-10.

---

## Dataset

* **Source:** `tf.keras.datasets.cifar10`
* **Input:**

  * Color images with shape 32 × 32 × 3.
* **Classes:**

  * 10 object categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.
* **Splits used in the notebook:**

  * Training: 45,000 images
  * Validation: 5,000 images
  * Test: 10,000 images
* **Preprocessing:**

  * Pixel values are scaled to the range [0, 1] by dividing by 255.
  * Labels are flattened with `.ravel()` so they have shape `(N,)`.

---

## What the Notebook Does

### 1. Libraries and Setup

* Imports:

  * numpy, pandas, matplotlib
  * tensorflow / tf.keras
  * sklearn.metrics (confusion_matrix, classification_report)
* Sets random seeds for reproducibility using `np.random.seed(100)` and `tf.random.set_seed(100)`.

### 2. Loading and Splitting CIFAR-10

* Loads training and test sets using `tf.keras.datasets.cifar10`.
* Splits the original training set into:

  * A validation set (first 5,000 images).
  * A reduced training set (remaining 45,000 images).
* Normalizes all images by dividing pixel values by 255.
* Flattens labels with `.ravel()` so they become 1D arrays.

### 3. Common Training Pipeline

Each trial in Experiment #12 follows this general pipeline:

1. **Clear graph and set seeds**

   * Uses `tf.keras.backend.clear_session()` and resets random seeds.

2. **Model definition**

   * `Flatten(input_shape=[32, 32, 3])` to turn images into vectors.
   * Several Dense layers with ReLU activations.
   * Dropout layers between hidden layers for regularization.
   * Final `Dense(10, activation='softmax')` for CIFAR-10 classification.

3. **Compile**

   * Optimizer: either SGD or Adam with a specified learning rate.
   * Loss: `SparseCategoricalCrossentropy(from_logits=False)`.
   * Metric: accuracy.

4. **Early stopping**

   * Uses `EarlyStopping` on validation loss with patience and `restore_best_weights=True`.

5. **Train and evaluate**

   * Trains on `(X_train, y_train)` and validates on `(X_valid, y_valid)`.
   * Evaluates on `(X_test, y_test)`.
   * Computes train and test accuracy using predicted labels.
   * Prints confusion matrix and classification report on the test set.
   * Plots training vs validation accuracy and loss over epochs.

---

## Experiments with MLPs on CIFAR-10

### Experiment #12 – First Trial (Baseline SGD MLP)

* **Architecture:**

  * Flatten
  * Dense(300, ReLU) + Dropout(0.15)
  * Dense(100, ReLU) + Dropout(0.15)
  * Dense(300, ReLU) + Dropout(0.15)
  * Dense(10, Softmax)
* **Optimizer:** SGD with learning rate 0.05.
* **Training setup:**

  * Up to 30 epochs.
  * Batch size: 64.
  * Early stopping on validation loss.
* **Purpose:**

  * Provide a baseline MLP performance on CIFAR-10 using SGD and moderate dropout.

### Experiment #12 – Second Trial (Deeper/Wider MLP with Adam)

* **Architecture:**

  * Flatten
  * Dense(512, ReLU) + Dropout(0.2)
  * Dense(256, ReLU) + Dropout(0.2)
  * Dense(128, ReLU) + Dropout(0.2)
  * Dense(64, ReLU) + Dropout(0.2)
  * Dense(10, Softmax)
* **Optimizer:** Adam with learning rate 0.001.
* **Training setup:**

  * Up to 10 epochs.
  * Batch size: 32.
  * Early stopping on validation loss.
* **Purpose:**

  * Test a deeper and wider architecture with Adam.
  * See whether more capacity and a different optimizer improve validation and test accuracy compared to the baseline SGD model.

### Experiment #12 – Third Trial (Refined Architecture with Smaller SGD Learning Rate)

* **Architecture:**

  * Flatten
  * Dense(350, ReLU) + Dropout(0.2)
  * Dense(150, ReLU) + Dropout(0.2)
  * Dense(350, ReLU) + Dropout(0.2)
  * Dense(10, Softmax)
* **Optimizer:** SGD with learning rate 0.025.
* **Training setup:**

  * Up to 30 epochs.
  * Batch size: 64.
  * Early stopping on validation loss.
* **Purpose:**

  * Strike a balance between the first and second trials:

    * Sufficient network capacity for CIFAR-10.
    * Dropout for regularization.
    * Smaller learning rate for more stable SGD training.
  * Compare training/validation curves and test accuracy with the other trials.

---

## Libraries Used

* Python 3.x
* numpy, pandas, matplotlib
* tensorflow / tf.keras

  * Sequential, Dense, Flatten, Dropout
  * Optimizers: SGD, Adam
  * EarlyStopping callback
* scikit-learn

  * confusion_matrix
  * classification_report

---

## How to Run

1. Open the notebook in Google Colab or a local Jupyter environment.
2. Make sure the following packages are installed:

   * tensorflow
   * scikit-learn
   * pandas
   * numpy
   * matplotlib
3. Run all cells in order:

   * CIFAR-10 will be downloaded automatically from `tf.keras.datasets.cifar10`.
4. Inspect the printed metrics and plots for all three trials of Experiment #12.

---

## Key Takeaways

* CIFAR-10 is significantly harder than MNIST, especially for plain MLPs on raw pixels.
* Wider and deeper MLPs with dropout can help but still struggle compared to convolutional neural networks (CNNs).
* Optimizer choice matters: Adam often converges faster, while SGD with a smaller learning rate can give more stable training.
* Architecture, dropout rate, learning rate, and batch size all strongly influence training stability and generalization.
* These experiments motivate moving beyond MLPs to CNNs for image classification tasks.

---
