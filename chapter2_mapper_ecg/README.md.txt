# Chapter 2: ECG Signal Analysis with Mapper

This folder contains the implementation and experiments corresponding to **Chapter 2** of my Bachelor's Thesis, focused on the application of the **Mapper algorithm** to physiological signals.

## Objective

The goal of this chapter is to explore whether **Topological Data Analysis (TDA)** can uncover meaningful patterns in ECG signals, potentially revealing **unknown subtypes of cardiac arrhythmias**.

Instead of analyzing signals individually, we study their **global structure** by representing them as a topological graph.

## Dataset

We work with the **Icentia11k dataset**, which contains continuous ECG recordings from over **11,000 patients**.

- Long-duration recordings (~70 minutes per segment)
- Beat-level and rhythm-level annotations
- Patients selected due to arrhythmias or suspected cardiac conditions

## Methodology

The workflow consists of several stages:

### 1. Preprocessing: before TDA, raw signals undergo a rigorous cleaning pipeline:
- Signal segmentation into fixed-length windows (25–30 seconds)
- Quality filtering and dataset reduction
- Feature extraction:
  - Statistical features (mean, std, skewness)
  - Frequency-domain features (LF, HF)
  - RR interval metrics
  - Signal amplitude features

### 2. Feature Representation
- Each window is transformed into a feature vector (up to 15 dimensions)
- Dimensionality reduction using **PCA** (to 10D)

### 3. Mapper Construction

Two complementary configurations are used:

####  Configuration A (Global Structure)
- Filter: **UMAP (2D projection)**
- Covering: 16×16 intervals, 50% overlap
- Clustering: **HDBSCAN**
- Goal: Capture the **overall organization** of ECG signals

####  Configuration B (Anomaly Detection)
- Filter: **Eccentricity (30-NN)**
- Projection: 1D
- Covering: 15 intervals, 50% overlap
- Clustering: **HDBSCAN**
- Goal: Highlight **rare or anomalous patterns**

## Why these lenses

The choice of the filter function (lens) determines the perspective of the topological map:

-  UMAP (Configuration A): Used to preserve the local manifold structure of the ECG features. It excels at separating well-defined cardiac regimes (e.g., normal sinus rhythm vs. persistent tachycardia).
-  Eccentricity (Configuration B): Defined as the distance to the "center" of the data cloud. High eccentricity nodes represent points that are far from the average behavior. This is our primary tool for Anomaly Detection, as it pushes irregular heartbeats to the "flares" or stalks of the graph.

## Analysis Techniques

### Graph Interpretation
Unlike traditional clustering, Mapper allows for overlapping clusters. 
- Nodes represent clusters of similar ECG windows
- Edges represent shared data points between clusters. Innn  this  Project, ann   Edge between two nodes represents a "Topological  Bridge". These bridges visualize the transition between cardiac states. A bridge between a "Normal" cluster and an "Anomalous" cluster represents heartbeats that are starting to show pathological signatures, providing a visual tool for early diagnosis.
- Global vs. local structures are analyzed separately

### Coloring Strategy
Graphs are enriched by coloring nodes according to:
- Physiological features (RR mean, RR std)
- Frequency-domain features (LF, HF, LF/HF ratio)
- Anomaly metrics:
  - kNN distance
  - Local Outlier Factor (LOF)
  - Eccentricity

This allows identifying:
- Transitions between cardiac regimes
- Regions of irregular behavior
- Candidate anomalous clusters

## Key Results

- Mapper successfully captures the **global topology** of ECG signals
- Clear **clusters and transitions** between different cardiac behaviors emerge
- Several **candidate anomalous nodes** are identified
- Expert evaluation shows that some detected anomalies correspond to:
  - Signal artifacts (e.g., noise, electrode issues)
  - Not necessarily new arrhythmias, but meaningful patterns

## Conclusion

The Mapper algorithm proves to be a powerful **exploratory tool** for ECG analysis:

- Reduces complexity of large datasets
- Provides interpretable graph-based representations
- Helps identify regions of interest for further clinical study

While no new arrhythmia subtype is definitively discovered, the method effectively narrows down **where to look**.

## Code & Dependencies

Main libraries used:
- `kmapper`
- `scikit-learn`
- `numpy`
- `matplotlib`
- `seaborn`

Refer to the code files in this folder for:
- Data preprocessing
- Feature extraction
- Mapper construction
- Visualization utilities

