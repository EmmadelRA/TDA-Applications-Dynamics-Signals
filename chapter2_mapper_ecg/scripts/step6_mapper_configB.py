import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
import umap
import hdbscan
import kmapper as km
import matplotlib.pyplot as plt
from scipy.stats import zscore
# Import our custom tools
from src.utils_mapper import (
    calculate_node_positions, 
    draw_mapper_graph, 
    calculate_knn_eccentricity, 
    get_node_stats,
    mapper_to_networkx
)

# Configuration 
INPUT_DATA = Path("data/mapper_input/X_mapper.npy")
FEATURE_NAMES = [
    "seg1_mean", "seg1_std", "seg1_skew",
    "seg2_mean", "seg2_std", "seg2_skew", 
    "seg3_mean", "seg3_std", "seg3_skew",
    "LF", "HF", "RR_mean", "RR_std", "amp_max", "amp_min"
]
REPORTS_DIR = Path("reports/config_b")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def run_mapper_anomalies():
    X = np.load(INPUT_DATA)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_pca = PCA(n_components=10, random_state=42).fit_transform(X_scaled)

    # 1. Generate 2D Layout for plotting
    umap_coords = umap.UMAP(n_neighbors=30, min_dist=0.05, random_state=42).fit_transform(X_pca)

    # 2. TDA Filter: kNN Eccentricity
    ecc = calculate_knn_eccentricity(X_pca, k=30)
    ecc_z = (ecc - np.mean(ecc)) / (np.std(ecc) + 1e-8)

    # 3. Build Mapper Graph
    mapper = km.KeplerMapper(verbose=1)
    cover = km.Cover(n_cubes=22, perc_overlap=0.5)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=8, min_samples=4)
    
    graph = mapper.map(ecc_z.reshape(-1, 1), X_pca, cover=cover, clusterer=clusterer)
    positions = calculate_node_positions(graph, umap_coords)

    # 4. Multidimensional Anomaly Metrics
    # kNN Distance
    nn = NearestNeighbors(n_neighbors=31).fit(X_pca)
    dists, _ = nn.kneighbors(X_pca)
    knn_dist = dists[:, 1:].mean(axis=1)
    
    # LOF Score
    lof = LocalOutlierFactor(n_neighbors=30)
    lof.fit(X_pca)
    lof_score = -lof.negative_outlier_factor_

    # Composite Rarity Score (Z-score mean)
    rarity_matrix = np.vstack([ecc_z, knn_dist, lof_score]).T
    composite_score = zscore(rarity_matrix, axis=0).mean(axis=1)

    # 5. Advanced Visualization (Integrated Map)
    # Features for node size and orange borders
    rr_std = X[:, FEATURE_NAMES.index("RR_std")]
    amp_range = X[:, FEATURE_NAMES.index("amp_max")] - X[:, FEATURE_NAMES.index("amp_min")]
    amp_threshold = np.percentile(amp_range, 95)

    node_rarity = get_node_stats(graph, composite_score)
    node_rr_std = get_node_stats(graph, rr_std)
    node_high_amp = get_node_stats(graph, (amp_range > amp_threshold).astype(float))

    # Save Integrated Plot
    draw_integrated_map(graph, positions, node_rarity, node_rr_std, node_high_amp)
    
    print(f"Analysis complete. Results saved in {REPORTS_DIR}")

def draw_integrated_map(graph, pos, rarity, size_metric, highlight_metric):
    import networkx as nx
    G = mapper_to_networkx(graph)
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Scale sizes for visibility
    s_vals = np.array([size_metric[n] for n in G.nodes()])
    node_sizes = 600 * (s_vals - s_vals.min()) / (s_vals.max() - s_vals.min() + 1e-8) + 100
    
    node_colors = [rarity[n] for n in G.nodes()]
    
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3)
    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, cmap="viridis", ax=ax)
    
    # Highlight high amplitude nodes with orange borders
    edge_nodes = [n for n in G.nodes() if highlight_metric[n] > 0.5]
    nx.draw_networkx_nodes(G, pos, nodelist=edge_nodes, 
                           node_size=node_sizes[[list(G.nodes()).index(n) for n in edge_nodes]],
                           node_color="none", edgecolors="orange", linewidths=2.5)

    plt.colorbar(nodes, ax=ax, label="Composite Rarity Score")
    ax.set_title("Config B: Integrated Anomaly Map\n(Color=Rarity, Size=RR_std, Border=High Amp)")
    plt.axis("off")
    plt.savefig(REPORTS_DIR / "integrated_anomaly_map.png", dpi=300)

if __name__ == "__main__":
    run_mapper_anomalies()

