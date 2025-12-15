# Neural Manifold Decoding from Synthetic EEG Data via Spectral PCA

## 1. Abstract
This project implements a complete data science pipeline to extract a low-dimensional topological subspace, known as a **Neural Manifold**, from noisy, high-dimensional electrophysiological data. By simulating the **Event-Related Desynchronization (ERD)** and **Synchronization (ERS)** characteristic of the motor cortex, we demonstrate that standardizing spectral features via Z-scoring, enables Principal Component Analysis (PCA) to recover latent dynamics, effectively separating signal covariance from the dominant background noise.

---

## 2. Neuroscience Background

### 2.1 The Manifold Hypothesis
The **Manifold Hypothesis** suggests that while the brain consists of billions of neurons, the neural activity related to specific tasks is constrained to a low-dimensional subspace. In the motor cortex, neural trajectories during reaching movements often exhibit rotational dynamics.

### 2.2 Motor Cortex Rhythms
The primary motor cortex (M1) exhibits distinct oscillatory signatures:
* **Alpha Rhythm (8–13 Hz):** The idling rhythm. High power indicates the motor cortex is at rest.
* **Beta Rhythm (13–30 Hz):** Associated with maintaining the current motor set.

### 2.3 ERD/ERS Dynamics
Movement execution follows a strict temporal pattern:
1.  **Planning Phase:** A sharp reduction in Alpha/Beta power, reflecting the release of inhibition.
2.  **Execution Phase:** Sustained desynchronization.
3.  **Rebound Phase:** A light post-movement surge in Beta power, reflecting active inhibition.

---

## 3. Mathematical Framework

### 3.1 Signal Generation
We model the continuous voltage signal $V_c(t)$ for channel $c$ as a superposition of a stochastic background process and deterministic oscillatory signals modulated by the task envelope $\mathcal{E}_{task}(t)$.

$$V_c(t) = \underbrace{N_{pink}(t)}_{\text{Background}} + \underbrace{\mathcal{E}_{task}(t) \cdot \left( A \sin(2\pi f_{\alpha} t) + B \sin(2\pi f_{\beta} t) \right)}_{\text{Motor Signal}} + \epsilon(t)$$

Where $N_{pink}(t)  \propto 1/f$ represents the background activity. This noise can be expressed as the sum of activity across all frequency bands $[\delta, \theta, \alpha, \beta, \gamma]$, heavily dominated by the lower frequencies:

$$N_{pink}(t) = \eta_{\delta}(t) + \eta_{\theta}(t) + \eta_{\alpha}(t) + \eta_{\beta}(t) + \eta_{\gamma}(t)$$

Here, $\eta_b(t)$ represents the stochastic noise component in band $b$, where $A(\eta_{\delta}) \gg A(\eta_{\gamma})$.

### 3.2 Spectral Analysis
Neural signals are **non-stationary**, meaning their frequency statistics change over time. We employ the **Short-Time Fourier Transform (STFT)** to map the time-series into a Time-Frequency representation.

For a discrete signal $x[n]$ sampled at a certain frequency, we apply a sliding window function $w[n]$ of length $L$ with a hop size $R$. The STFT at time frame $m$ and frequency bin $k$ is defined as:

$$X[m, k] = \sum_{n=0}^{L-1} x[n + mR] \cdot w[n] \cdot e^{-j \frac{2\pi}{L} k n}$$

The **Power Spectral Density (PSD)**, or spectrogram, is the squared magnitude of the complex coefficients:

$$S[m, k] = |X[m, k]|^2$$

To extract neurophysiologically relevant features, we average the power over specific frequency bands $b = [k_{0}, k_{end}]$:

$$P_{b}[m] = \frac{1}{N_k} \sum_{k=k_{0}}^{k_{end}} S[m, k]$$


### 3.3 Standardization
EEG data follows a power law ($P \propto 1/f$), meaning low-frequency variance dominates the signal:

$$\sigma^2_{\delta} \gg \sigma^2_{\theta} \gg \sigma^2_{\alpha} \gg \sigma^2_{\beta} \gg \sigma^2_{\gamma}$$

Applying PCA directly to raw power values would result in PC1 merely tracking the random drift of the Delta band. To recover the information structure, namely the synchrony, rather than amplitude, we **standardize** the features column-wise via Z-scoring.

Given a feature matrix $\mathbf{X} \in \mathbb{R}^{N \times F}$, where $N$ is total time points across all trials:

$$\mathbf{Z}_{ij} = \frac{\mathbf{X}_{ij} - \mu_j}{\sigma_j}$$

Where $\mu_j$ and $\sigma_j$ are the mean and standard deviation of the $j$-th frequency band. This ensures all bands have unit variance ($\sigma^2=1$), effectively whitening the spectrum.

### 3.4 Dimensionality Reduction
We identify the Neural Manifold using **Principal Component Analysis (PCA)** on the standardized features.

**Correlation Matrix**: We compute the covariance matrix of the standardized data $\mathbf{Z}$, which is equivalent to the correlation matrix of the original data $\mathbf{X}$:

$$
\mathbf{C} = \frac{1}{N-1} \mathbf{Z}^\top \mathbf{Z} \in \mathbb{R}^{F \times F}
$$

**Eigendecomposition**: We solve for the eigenvalues $\lambda$ and eigenvectors $\mathbf{v}$:

$$\mathbf{C} \mathbf{v}_k = \lambda_k \mathbf{v}_k$$

Here, the eigenvectors $\mathbf{V} = [\mathbf{v}_1, \dots, \mathbf{v}_F]$ represent the **principal axes** of the neural manifold, and the eigenvalues $\lambda_k$ represent the variance explained by each axis.

**Projection**: The low-dimensional neural trajectory $\mathbf{T}$ is obtained by projecting the standardized data onto the top $d$ principal components:

$$\mathbf{T} = \mathbf{Z} \cdot \mathbf{W}_d$$

```math
\begin{bmatrix} t_{1,1} & \dots & t_{1,d} \\ \vdots & \ddots & \vdots \\ t_{N,1} & \dots & t_{N,d} \end{bmatrix} = \begin{bmatrix} z_{1,1} & \dots & z_{1,F} \\ \vdots & \ddots & \vdots \\ z_{N,1} & \dots & z_{N,F} \end{bmatrix} \cdot \begin{bmatrix} | & & | \\ \mathbf{v}_1 & \dots & \mathbf{v}_d \\ | & & | \end{bmatrix}
```

---

## 4. Pipeline Architecture

| Step | Process | Tensor Transformation | Dimensions |
| :--- | :--- | :--- | :--- |
| **1** | **Signal Generation** | `(Trials, Channels, Time)` | $(100, 2, 12000)$ |
| **2** | **Spectral Analysis** | `(Trials, Windows, Features)` | $(100, 60, 10)$ |
| **3** | **Standardization** | `(TotalTimePoints, Features)` | $(6000, 10)$ |
| **4** | **Dimensionality Reduction** | `(Trials, Windows, Features)` | $(100, 60, 3)$ |
---

## Getting Started

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
---

## Project Structure

### Notebooks
* `notebooks/01_signal_generation.ipynb`: Generates the synthetic EEG data.
* `notebooks/02_spectral_analysis.ipynb`: Performs the spectral analysis.
* `notebooks/03_manifold_decoding.ipynb`: Applies Z-scoring and PCA.

### Source Code
* `src/generation.py`: Physics engine for simulating the non-stationary signals.
* `src/processing.py`: Utilities for STFT and spectral power integration.

### Configuration & Data
* `requirements.txt`: List of the Python dependencies.
* `config.yaml`: Central parameter configuration.
* `data/`: Directory structure for storing assets.
