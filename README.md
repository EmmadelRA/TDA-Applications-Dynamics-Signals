# Applications of Topological Data Analysis

This repository contains the technical implementation and research developed for my **Bachelor's Thesis (TFG) in Mathematics** at the University of Zaragoza.

The project explores how tools from Algebraic Topology can be used to extract meaningful patterns from complex datasets, focusing on two main tools: **Mapper algorithm** and **Persistent Homology** applied to two different areas: **Biological Signal Analysis** and **Dynamical Systems**.

## Repository Structure:

The work is divided into two independent chapters, each with its own source code and experiments:

### 1. [Chapter 2: ECG Signal Analysis with Mapper](./chapter2_mapper_ecg/)
* **Focus:** Analyze how ECG signals cluster based on their global topological properties to identify potentially unknown subtypes of cardiac pathologies.
* **Methodology:** Using the **Mapper Algorithm** to create topological graphs that represent the "shape" of cardiac cycles.
* **Key results:** Identification of clusters and transitions in healthy vs. pathological signals.

### 2. [Chapter 3: Chaos Detection with TDA + CNN](./chapter3_chaos_tda_cnn/)
* **Focus:** Classification of chaotic vs. regular behavior in dynamical systems.
* **Methodology:** * State-space reconstruction via **Takens' Embedding** for logistic map and direct 3D representation for Lorenz System.
    * Computation of **Persistence Diagrams** and vectorization to **Persistence Images**.
    * Classification using a custom **Convolutional Neural Network (CNN)**.
* **Systems:** Logistic Map and Lorenz System (3D).

## Installation & Requirements
This project requires Python 3.8+ and the following core libraries:
`numpy`, `torch`, `ripser`, `persim`, `kmapper`, `scikit-learn`, `matplotlib`, `seaborn`.

Detailed installation instructions and specific dependencies can be found within each chapter folder.

## Academic Information
* **Author:** Emma Del Río Angulo
* **Institution:** Facultad de Ciencias, Universidad de Zaragoza.
* **Advisor:** Rubén Vigara Benito
* **Date:** November 2025

