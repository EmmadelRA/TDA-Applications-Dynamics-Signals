# Chapter 3: Chaos Detection with TDA + CNN

This folder contains the implementation of **Chapter 3** of my Bachelor's Thesis, where **Topological Data Analysis (TDA)** is combined with **Deep Learning** to classify dynamical systems.

## Objective

The goal is to determine whether **persistent homology** can distinguish between:

- **Chaotic behavior**
- **Regular (non-chaotic) behavior**

and to automate this classification using a **Convolutional Neural Network (CNN)**.

## Dynamical Systems Studied

Two classical systems are analyzed:

### Lorenz System (Continuous, 3D)
- Generates trajectories directly in ℝ³
- Known for its chaotic attractor

### Logistic Map (Discrete, 1D)
- Defined by:  
  \( x_{n+1} = r x_n (1 - x_n) \)
- Exhibits both periodic and chaotic regimes depending on parameter \( r \)

## Methodology

### 1. Data Generation

- Simulate trajectories for both systems under different parameter regimes
- Label each trajectory as **chaotic** or **non-chaotic**

### 2. State-Space Reconstruction

- **Lorenz system:** direct 3D trajectories
- **Logistic map:** reconstructed using **Takens' Embedding**
  - Embedding dimension \( d \)
  - Time delay \( \tau \)

This transforms time series into **point clouds in ℝᵈ**

### 3. Persistent Homology

- Build **Vietoris–Rips filtrations**
- Compute persistence diagrams using:
  - `ripser`
- Analyze topological features:
  - \( H_0 \): connected components
  - \( H_1 \): loops
  - \( H_2 \): higher-dimensional cavities

### 4. Vectorization

- Convert persistence diagrams into **Persistence Images**
- This step makes topological information compatible with ML models

### 5. Classification with CNN

- Input: persistence images
- Model: custom **Convolutional Neural Network (CNN)** implemented in PyTorch
- Output: binary classification (chaotic vs non-chaotic)

## Key Observations

- Chaotic systems produce:
  - More **dispersed persistence diagrams**
  - More **long-lived topological features**
  - Greater presence of higher-dimensional features

- Non-chaotic systems show:
  - Simpler, more concentrated diagrams
  - Fewer persistent features

## Results

- Classification accuracy exceeds **99%** for both systems
- Misclassifications occur mainly near:
  - **Transition regions** between chaos and regularity

## Comparison with Literature

- Results are consistent with prior work
- In some cases, performance is comparable or slightly improved
- Confirms that TDA + CNN is a viable alternative to:
  - Classical chaos indicators (e.g., Lyapunov exponents)

## Conclusion

This chapter demonstrates that:

- Persistent homology captures **structural complexity** of dynamical systems
- Topological features can be successfully integrated into **deep learning pipelines**
- The approach is:
  - Robust
  - Interpretable
  - Highly accurate

## Code & Dependencies

Main libraries used:
- `ripser`
- `persim`
- `numpy`
- `scikit-learn`
- `torch`

Contents of this folder include:
- Simulation of dynamical systems
- Takens embedding implementation
- Persistence diagram computation
- Persistence image generation
- CNN training and evaluation