# Neural Manifold Decoding from Synthetic EEG Data via Spectral PCA

**A computational neuroscience pipeline to simulate non-stationary field potentials and recover latent low-dimensional dynamics.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

## 📌 Overview

This project demonstrates the extraction of **Neural Manifolds**—low-dimensional topological subspaces where neural population activity evolves—from noisy, high-dimensional EEG data.

Neural field potentials are dominated by **1/f background noise** (pink noise), often obscuring the task-relevant signals in specific oscillatory bands. This repository implements a complete data science pipeline to validate that by **standardizing (Z-scoring) spectral features**, unsupervised learning methods like Principal Component Analysis (PCA) can successfully identify signal covariance over noise variance. This allows for the recovery of the latent "butterfly" trajectory characteristic of motor cortex dynamics during reaching tasks.

---

## 🧠 Neuroscience Background

The simulation is grounded in established neurophysiology of the mammalian motor cortex.

### 1. The Idling Rhythm Hypothesis
The primary motor cortex (M1) is not silent when at rest. Instead, large populations of neurons fire in synchrony, generating strong oscillatory electrical fields.
* **Mu/Alpha Rhythm (8–13 Hz):** The dominant idling rhythm of the sensorimotor system.
* **Beta Rhythm (13–30 Hz):** Associated with maintaining the current motor set (status quo).

### 2. Event-Related Desynchronization (ERD)
When a movement is planned or executed, these synchronous populations "break apart" to process information independently. This results in a massive **drop in amplitude** (power) in the Alpha/Beta bands. This phenomenon is known as **ERD**.

### 3. Event-Related Synchronization (ERS) / Beta Rebound
Immediately after movement cessation, the cortex "resets" with an inhibitory rebound. The Beta power not only returns to baseline but briefly overshoots it. This is the **Post-Movement Beta Rebound**.

### 4. Contralateral Lateralization
The brain controls the body cross-wise.
* **Left Hand Movement** $\rightarrow$ ERD in the **Right** Motor Cortex (Channel C4).
* **Right Hand Movement** $\rightarrow$ ERD in the **Left** Motor Cortex (Channel C3).

---

## 🧮 Mathematical Framework & Data Science Techniques

### 1. Signal Generation (The Physics)
We model the raw voltage $V(t)$ for a given channel as a superposition of stochastic noise and deterministic oscillatory signals modulated by a task envelope.

$$V(t) = N_{pink}(t) + A(t) \cdot \sin(2\pi f_{\alpha} t) + B(t) \cdot \sin(2\pi f_{\beta} t) + \epsilon(t)$$

Where:
* $N_{pink}(t) \propto 1/f^{\gamma}$: Pink noise representing background synaptic activity (high power at low frequencies).
* $A(t), B(t)$: Time-varying envelopes representing the cognitive state (Rest $\to$ ERD $\to$ ERS).
* $\epsilon(t)$: Sensor white noise.

### 2. Spectral Analysis (STFT)
Since the signal is **non-stationary** (the frequency content changes over time), simple Fourier Transform (DFT) is insufficient. We use the **Short-Time Fourier Transform (STFT)** to map voltage to the time-frequency domain.

For a discrete signal $x[n]$ and window $w[n]$:

$$X(m, \omega) = \sum_{n=-\infty}^{\infty} x[n] w[n-mR] e^{-j\omega n}$$

This transforms our data dimensions from Time to Time-Frequency.

### 3. Feature Engineering (Band Power)
We integrate the power spectral density (PSD) over specific neurophysiologically relevant bands to create features. For a band $b \in [\omega_{low}, \omega_{high}]$:

$$P_b[m] = \frac{1}{N_{bins}} \sum_{\omega=\omega_{low}}^{\omega_{high}} |X(m, \omega)|^2$$

### 4. Standardization (Z-Scoring)
This is the **critical step**. EEG data follows a power law ($P \propto 1/f$), meaning Delta band variance ($\sigma^2_{\delta}$) is orders of magnitude larger than Beta band variance ($\sigma^2_{\beta}$).
If we applied PCA directly, PC1 would purely capture the random Delta drift.

We standardize every feature column $j$:
$$Z_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}$$
This forces $\sigma^2 = 1$ for all bands, converting the PCA problem from analyzing the **Covariance Matrix** (amplitude-driven) to the **Correlation Matrix** (synchrony-driven).

### 5. Dimensionality Reduction (PCA)
We use Principal Component Analysis to find the neural manifold. PCA finds the eigenvectors (principal components) of the correlation matrix $C = Z^T Z$.

$$C = V \Lambda V^T$$

* **PC1:** Captures the shared ERD (movement state).
* **PC2:** Captures the lateralization (Left vs. Right hand).
* **PC3:** Captures the rotational dynamics (Rebound phase).

---

## 📊 Pipeline & Data Dimensions

The data undergoes a sequence of transformations, reshaping the tensor at each step.

### Step 1: Raw Data Generation
We simulate 20 trials (10 Left, 10 Right) of 6 seconds each at 250 Hz.
* **Input:** Simulation Parameters.
* **Output Data Structure:** 3D Tensor.
    $$X_{raw} \in \mathbb{R}^{N_{trials} \times N_{channels} \times N_{time}}$$
    $$\text{Dimensions: } (20 \times 2 \times 1500)$$

### Step 2: Spectral Feature Extraction
We compute the STFT and average into 5 bands ($\delta, \theta, \alpha, \beta, \gamma$) per channel.
* **Transformation:** Voltage ($V$) $\rightarrow$ Power ($dB$).
* **Intermediate Tensor:**
    $$X_{features} \in \mathbb{R}^{N_{trials} \times N_{channels} \times N_{bands} \times N_{windows}}$$
    $$\text{Dimensions: } (20 \times 2 \times 5 \times 60)$$
* **Flattening:** We combine channels and bands into a single "Feature Vector" of size 10 (2 channels $\times$ 5 bands).
    $$X_{flat} \in \mathbb{R}^{N_{trials} \times N_{windows} \times N_{features}}$$
    $$\text{Dimensions: } (20 \times 60 \times 10)$$

### Step 3: Manifold Learning (PCA)
We stack all trials vertically to treat every time window as an independent observation of the brain state.
* **Stacked Matrix:**
    $$X_{stacked} \in \mathbb{R}^{N_{total\_samples} \times N_{features}}$$
    $$\text{Dimensions: } (1200 \times 10)$$
* **PCA Transformation:** We project this 10D space into a 3D latent space.
    $$T = X_{stacked} \cdot W_{pca}$$
    $$T \in \mathbb{R}^{1200 \times 3}$$

---

## 📂 Project Structure

### 📓 Notebooks (Analysis Pipeline)
* **`notebooks/01_signal_generation.ipynb`**:
    * *Goal*: Generates the synthetic dataset (20 trials, 2 channels).
    * *Details*: Implements the random walk (Brownian noise) for Delta bands and the ERD envelopes for Alpha/Beta.
* **`notebooks/02_spectral_analysis.ipynb`**:
    * *Goal*: Performs STFT and extracts band-power features.
    * *Details*: Reshapes the 3D raw tensor into the 2D feature matrix required for scikit-learn.
* **`notebooks/03_manifold_decoding.ipynb`**:
    * *Goal*: Applies Z-scoring and PCA to uncover latent dynamics.
    * *Details*: Visualizes the 3D manifold, the covariance matrix, and the PCA loading vectors to validate the "Signal vs. Noise" separation.

### 🛠 Source Code
* **`src/generation.py`**:
    * Contains the physics engine for generating pink noise, artifacts, and motor-task modulation envelopes.
* **`src/processing.py`**:
    * Utility functions for signal processing, including STFT computation and band-power averaging.

### 💾 Data Directory
* `data/01_raw/`: Stores generated raw voltage time-series (`.npy`).
* `data/02_features/`: Stores processed spectral feature matrices (`.npy`).
* `data/03_results/`: Stores final PCA coordinates and manifold plots.

### ⚙️ Configuration
* `config.yaml`: Central configuration file for simulation parameters.
* `requirements.txt`: List of Python dependencies.

---

## 🚀 Installation

### Prerequisites
* Python 3.8+
* Jupyter Notebook

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/mattia-3rne/neural-manifold-decoding-via-spectral-pca.git
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Pipeline**:
    Open the notebooks in order:
    1.  `01_signal_generation.ipynb`
    2.  `02_spectral_analysis.ipynb`
    3.  `03_manifold_decoding.ipynb`