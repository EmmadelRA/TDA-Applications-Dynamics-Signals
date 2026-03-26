import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap
import hdbscan
import kmapper as km
from src.utils_mapper import calculate_node_positions, draw_mapper_graph, get_node_means

# Configuration
INPUT_DATA = Path("data/mapper_input/X_mapper.npy")
FEATURE_NAMES = [
    "seg1_mean", "seg1_std", "seg1_skew",
    "seg2_mean", "seg2_std", "seg2_skew", 
    "seg3_mean", "seg3_std", "seg3_skew",
    "LF", "HF", "RR_mean", "RR_std", "amp_max", "amp_min"
]

def run_mapper_config_a():
    X = np.load(INPUT_DATA)
    
    # 1. Dimensionality Reduction
    scaler = StandardScaler()
    X_pca = PCA(n_components=10).fit_transform(scaler.fit_transform(X))
    
    # 2. Filter Function (Lens): UMAP 2D
    lens = umap.UMAP(n_neighbors=80, min_dist=0.1, n_components=2).fit_transform(X_pca)
    
    # 3. KeplerMapper Construction
    mapper = km.KeplerMapper(verbose=1)
    cover = km.Cover(n_cubes=14, perc_overlap=0.5)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=40, min_samples=5)
    
    graph = mapper.map(lens, X_pca, cover=cover, clusterer=clusterer)
    
    # 4. Visualization & Analysis
    positions = calculate_node_positions(graph, lens)
    
    # Generate Base Visualization
    draw_mapper_graph(graph, positions, title="Config A: Global Structure", output_file="reports/A_base.png")
    
    # Generate Feature-specific Visualizations (Example: RR_mean)
    rr_mean_idx = FEATURE_NAMES.index("RR_mean")
    node_stats = get_node_means(graph, X[:, rr_mean_idx])
    draw_mapper_graph(graph, positions, node_values=node_stats, 
                      title="Config A: Heart Rate (RR Mean)", output_file="reports/A_RR_mean.png")
    
    print("Mapper Analysis for Config A completed")

if __name__ == "__main__":
    run_mapper_config_a()
