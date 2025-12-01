

---

## README for `Simeon_Bright_assig3_PartA.ipynb`

````markdown
# Assignment 3 – Multilayer Perceptrons on MNIST (Part A)

## Overview

This notebook trains a sequence of **multilayer perceptron (MLP)** models on the **MNIST handwritten digits** dataset. The focus is to:

- Compare different **weight initializers** (zeros, ones, random normal).
- Explore **depth** (2, 3, 10, 20 hidden layers).
- Study **dropout** placement and strength.
- Observe the effect of **learning rate** on convergence and overfitting.
- Use early stopping, confusion matrices and learning curves to analyze performance.

The notebook is organized into **Experiments #1–11**, with an additional **“Experiment #11 New Trial”** to reduce overfitting.

---

## Dataset

- **Source:** `tf.keras.datasets.mnist`
- **Input:**
  - Grayscale images of size **28 × 28**.
- **Classes:** digits `0–9` (10 classes).
- **Splits used:**
  - Training: 55,000 images  
  - Validation: 5,000 images  
  - Test: 10,000 images
- **Preprocessing:**
  - Pixel intensities are scaled to `[0, 1]` by dividing by 255:
    ```python
    X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
    X_test = X_test / 255.0
    ```

---

## What the Notebook Does

### 1. Libraries and Setup

- Imports:
  - `numpy`, `pandas`, `matplotlib`, `time`, `warnings`
  - `tensorflow` / `tf.keras` (Sequential, Dense, Flatten, optimizers, callbacks)
  - `sklearn.metrics` (`confusion_matrix`, `classification_report`)
- Sets random seeds for reproducibility:
  - `np.random.seed(100)`, `tf.random.set_seed(100)`.

### 2. Loading and Scaling MNIST

- Loads the MNIST dataset:
  ```python
  mnist = tf.keras.datasets.mnist
  (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()
````

* Splits off a validation set from the original training set.
* Normalizes images by dividing by 255.

### 3. Common Training Pipeline

Each experiment follows a similar training pattern:

1. **Reset the graph:**

   * `tf.keras.backend.clear_session()` to avoid graph buildup.
2. **Build the model:**

   * `Flatten(input_shape=[28, 28])`.
   * Several `Dense` layers with ReLU activations.
   * Optional `Dropout` layers between hidden layers.
   * Final `Dense(10, activation='softmax')`.
3. **Compile:**

   * Optimizer: `SGD(learning_rate=...)`.
   * Loss: `SparseCategoricalCrossentropy(from_logits=False)`.
   * Metrics: accuracy.
4. **Early stopping:**

   * `EarlyStopping` on validation loss with patience and `restore_best_weights=True`.
5. **Evaluate and analyze:**

   * Print train and test loss/accuracy.
   * Compute train/test accuracy manually from predicted labels.
   * Show **confusion matrix** and **classification report**.
   * Plot **training vs validation accuracy** and **loss** across epochs.

---

## Experiments with MLPs on MNIST

### Experiment #1 – 2-layer MLP with Zero Weights

* **Initializer:** `kernel_initializer = 'zeros'`.
* **Architecture:**

  * Flatten
  * Dense(300, ReLU)
  * Dense(100, ReLU)
  * Dense(10, Softmax)
* **Optimizer:** `SGD(learning_rate=0.1)`.
* Purpose: illustrate what happens when all weights start at zero (symmetry).

### Experiment #2 – 2-layer MLP with Ones Initialization

* **Initializer:** `kernel_initializer = 'ones'`.
* **Architecture:** same 2-layer MLP (300 → 100 → 10).
* **Optimizer:** `SGD(learning_rate=0.1)`.
* Purpose: compare training dynamics when all weights start at **one** rather than zero.

### Experiment #3 – 2-layer MLP with Random Normal Initialization

* **Initializer:** `kernel_initializer = 'random_normal'`.
* **Architecture:** same 2-layer MLP (300 → 100 → 10).
* **Optimizer:** `SGD(learning_rate=0.1)`.
* Purpose: move to a more realistic initialization strategy and see improved optimization.

### Experiment #4 – Dropout on First Hidden Layer (Strong Dropout)

* **Initializer:** `random_normal`.
* **Architecture:**

  * Flatten
  * Dense(300, ReLU)
  * Dropout(rate=0.5) after first hidden layer
  * Dense(100, ReLU)
  * Dense(10, Softmax)
* **Optimizer:** `SGD(learning_rate=0.1)`.
* Purpose: test strong dropout on the first hidden layer and observe impact on overfitting.

### Experiment #5 – 2-layer Baseline (Random Normal, No Dropout)

* **Initializer:** `random_normal`.
* **Architecture:** 300 → 100 → 10 (no dropout).
* **Optimizer:** `SGD(learning_rate=0.1)`, up to 50 epochs.
* Purpose: establish a **clean baseline** with realistic initialization and no dropout.

### Experiment #6 – Uniform Dropout (0.1 on Both Hidden Layers)

* **Initializer:** `random_normal`.
* **Architecture:**

  * Dense(300, ReLU) + Dropout(0.1)
  * Dense(100, ReLU) + Dropout(0.1)
  * Dense(10, Softmax)
* **Optimizer:** `SGD(learning_rate=0.1)`.
* Purpose: compare light dropout on both hidden layers to the no-dropout baseline.

### Experiment #7 – Asymmetric Dropout (0.5 then 0.1)

* **Initializer:** `random_normal`.
* **Architecture:**

  * Dense(300, ReLU) + Dropout(0.5)
  * Dense(100, ReLU) + Dropout(0.1)
  * Dense(10, Softmax)
* **Optimizer:** `SGD(learning_rate=0.1)`.
* Purpose: test **stronger dropout** on the first layer and lighter on the second.

### Experiment #8 – High Learning Rate with Dropout

* **Initializer:** `random_normal`.
* **Architecture:** same as Experiment #7 (300 → 100 with Dropout 0.5 / 0.1).
* **Optimizer:** `SGD(learning_rate=0.5)` (much larger).
* Purpose: examine how a large learning rate interacts with dropout in terms of convergence and stability.

### Experiment #9 – 10 Hidden Layers (100 Units Each)

* **Initializer:** `random_normal`.
* **Architecture:**

  * Flatten
  * Loop: 10 × [Dense(100, ReLU)]
  * Dense(10, Softmax)
* **Optimizer:** `SGD(learning_rate=0.1)`.
* Purpose: scale depth to 10 layers and study optimization difficulty and potential overfitting.

### Experiment #10 – 20 Hidden Layers (100 Units Each)

* **Initializer:** `random_normal`.
* **Architecture:**

  * Flatten
  * Loop: 20 × [Dense(100, ReLU)]
  * Dense(10, Softmax)
* **Optimizer:** `SGD(learning_rate=0.1)`.
* Purpose: push depth even further and compare training/validation curves to the 10-layer case.

### Experiment #11 – 4 Hidden Layers with Dropout 0.1

* **Initializer:** `random_normal`.
* **Architecture:**

  * Flatten
  * Dense(100, ReLU) + Dropout(0.1)
  * Dense(300, ReLU) + Dropout(0.1)
  * Dense(300, ReLU) + Dropout(0.1)
  * Dense(100, ReLU) + Dropout(0.1)
  * Dense(10, Softmax)
* **Optimizer:** `SGD(learning_rate=0.1)`.
* Purpose: build a **moderately deep** but symmetric MLP and see if a small dropout rate is enough to control overfitting.

### Experiment #11 – New Trial (Solving Overfitting)

* **Architecture:** same 4-hidden-layer model as Experiment #11.
* **Changes:**

  * Increase dropout to `rate=0.15`.
  * Reduce learning rate to `0.05`.
* Purpose: explicitly address overfitting seen in the previous run by:

  * Strengthening regularization (more dropout).
  * Slowing down learning (smaller LR).
* Evaluation uses the same metrics (accuracy, confusion matrix, learning curves).

---

## Libraries Used

* Python 3.x
* `numpy`, `pandas`, `matplotlib`
* `tensorflow` / `tf.keras`

  * `Sequential`, `Dense`, `Flatten`, `Dropout`
  * `optimizers.SGD`
  * `callbacks.EarlyStopping`
* `scikit-learn`

  * `confusion_matrix`
  * `classification_report`

---

## How to Run

1. Open the notebook in **Google Colab** or local Jupyter.
2. Make sure the following packages are installed:

   ```bash
   pip install tensorflow scikit-learn pandas numpy matplotlib
   ```
3. Run all cells in order:

   * MNIST will be downloaded automatically via `tf.keras.datasets.mnist`.
4. View the printed metrics and plots for each experiment.

---

## Key Takeaways

* Weight initialization (zeros, ones, random) has a big impact on whether training works at all.
* Moderate-depth MLPs (2–4 hidden layers) can reach strong performance on MNIST.
* Simply making the network very deep (10–20 layers) can hurt training without additional tricks.
* **Dropout** and **learning rate** are powerful levers for controlling overfitting and training stability.
* Early stopping helps select a good epoch without manual tuning.

````

-----------------------------------------------------------------------------------

## README for `Simeon_Bright_assig3_PartB.ipynb`

```markdown
# Assignment 3 – Multilayer Perceptrons on CIFAR-10 (Part B)

## Overview

This notebook applies **multilayer perceptron (MLP)** models to the **CIFAR-10** color image dataset. It extends Assignment 3 Part A from MNIST to a more challenging, high-dimensional problem.

The goals are to:

- Build fully-connected networks for **10-class image classification**.
- Compare different **architectures** (width/depth).
- Explore **dropout** and **learning rate** choices.
- Compare **SGD** vs **Adam** optimizers.
- Evaluate performance using accuracy, confusion matrices, and learning curves.

All three models are grouped under **Experiment #12**, as three different trials on CIFAR-10.

---

## Dataset

- **Source:** `tf.keras.datasets.cifar10`
- **Input:**
  - Color images with shape **32 × 32 × 3**.
- **Classes:**
  - 10 object categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.
- **Splits used in the notebook:**
  - Training: 45,000 images  
  - Validation: 5,000 images  
  - Test: 10,000 images
- **Preprocessing:**
  - Pixel values are scaled to `[0, 1]` by dividing by 255.
  - Labels are flattened with `.ravel()` so they have shape `(N,)`.

---

## What the Notebook Does

### 1. Libraries and Setup

- Imports:
  - `numpy`, `pandas`, `matplotlib`
  - `tensorflow` / `tf.keras`
  - `sklearn.metrics` (`confusion_matrix`, `classification_report`)
- Sets random seeds for reproducibility (`np.random.seed(100)` and `tf.random.set_seed(100)`).

### 2. Loading and Splitting CIFAR-10

- Loads training and test sets:
  ```python
  cifar10 = tf.keras.datasets.cifar10
  (X_train_full, y_train_full), (X_test, y_test) = cifar10.load_data()
````

* Creates validation split:

  ```python
  X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
  y_valid, y_train = y_train_full[:5000].ravel(), y_train_full[5000:].ravel()
  X_test = X_test / 255.0
  y_test = y_test.ravel()
  ```
* Prints shapes and dtypes of the splits.

### 3. Common Training Pipeline

Each trial in Experiment #12 shares a core structure:

1. **Clear graph and set seeds.**
2. **Model definition:**

   * `Flatten(input_shape=[32, 32, 3])`.
   * Several `Dense` layers with ReLU activations.
   * `Dropout` layers between hidden layers to reduce overfitting.
   * Final `Dense(10, activation='softmax')`.
3. **Compile:**

   * Optimizer: either `SGD` or `Adam` with a specified `learning_rate`.
   * Loss: `SparseCategoricalCrossentropy(from_logits=False)`.
   * Metric: accuracy.
4. **Early stopping:**

   * `EarlyStopping` on validation loss with patience, restoring best weights.
5. **Train and evaluate:**

   * Fit on `(X_train, y_train)` with validation on `(X_valid, y_valid)`.
   * Evaluate on `(X_test, y_test)`.
   * Compute train/test accuracy using predicted labels.
   * Print **confusion matrix** and **classification report**.
   * Plot training vs validation **accuracy** and **loss**.

----

## Experiments with MLPs on CIFAR-10

### Experiment #12 – First Trial (Baseline SGD MLP)

* **Architecture:**

  * Flatten
  * Dense(300, ReLU) → Dropout(0.15)
  * Dense(100, ReLU) → Dropout(0.15)
  * Dense(300, ReLU) → Dropout(0.15)
  * Dense(10, Softmax)
* **Optimizer:** `SGD(learning_rate=0.05)`.
* **Training setup:**

  * Up to 30 epochs.
  * Batch size: 64.
  * Early stopping on validation loss.
* **Purpose:**

  * Provide a **baseline MLP** performance on CIFAR-10 using SGD and moderate dropout.

### Experiment #12 – Second Trial (ChatGPT-Assisted Deeper/Wider MLP with Adam)

* **Architecture:**

  * Flatten
  * Dense(512, ReLU) → Dropout(0.2)
  * Dense(256, ReLU) → Dropout(0.2)
  * Dense(128, ReLU) → Dropout(0.2)
  * Dense(64, ReLU) → Dropout(0.2)
  * Dense(10, Softmax)
* **Optimizer:** `Adam(learning_rate=0.001)`.
* **Training setup:**

  * Up to 10 epochs.
  * Batch size: 32.
  * Early stopping enabled.
* **Purpose:**

  * Test a **deeper and wider** architecture suggested with ChatGPT help.
  * See whether Adam plus more capacity improves validation and test accuracy compared to the baseline SGD model.

### Experiment #12 – Third Trial (Refined Architecture with Smaller SGD Learning Rate)

* **Architecture:**

  * Flatten
  * Dense(350, ReLU) → Dropout(0.2)
  * Dense(150, ReLU) → Dropout(0.2)
  * Dense(350, ReLU) → Dropout(0.2)
  * Dense(10, Softmax)
* **Optimizer:** `SGD(learning_rate=0.025)`.
* **Training setup:**

  * Up to 30 epochs.
  * Batch size: 64.
  * Early stopping on validation loss.
* **Purpose:**

  * Strike a **balance** between the first and second trials:

    * Enough depth and width for CIFAR-10.
    * Dropout to regularize.
    * Smaller learning rate for more stable SGD training.
  * Compare training/validation curves and test accuracy against the previous two setups.

---

## Libraries Used

* Python 3.x
* `numpy`, `pandas`, `matplotlib`
* `tensorflow` / `tf.keras`

  * `Sequential`, `Dense`, `Flatten`, `Dropout`
  * Optimizers: `SGD`, `Adam`
  * `callbacks.EarlyStopping`
* `scikit-learn`

  * `confusion_matrix`
  * `classification_report`

---

## How to Run

1. Open the notebook in **Google Colab** or a local Jupyter environment.
2. Install required packages if needed:

   ```bash
   pip install tensorflow scikit-learn pandas numpy matplotlib
   ```
3. Run all cells from top to bottom:

   * CIFAR-10 will be downloaded automatically via `tf.keras.datasets.cifar10`.
4. Inspect the printed metrics and plots for all three trials of Experiment #12.

---

## Key Takeaways

* CIFAR-10 is significantly harder than MNIST, especially for **plain MLPs on raw pixels**.
* Wider/deeper MLPs with **dropout** can improve performance but still struggle compared to CNNs.
* **Optimizer choice** matters: Adam often converges faster, while SGD with a smaller learning rate can be more stable.
* Architecture, dropout rate, learning rate, and batch size all interact to determine training stability and generalization.
* These experiments motivate moving beyond MLPs to **convolutional neural networks** for image classification tasks.

```

If you upload the next notebook, I’ll keep using **this exact template** so all your README files look consistent in your GitHub repo.
```
