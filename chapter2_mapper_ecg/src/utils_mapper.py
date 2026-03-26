import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.neighbors import NearestNeighbors

def calculate_node_positions(graph_dict, lens_2d):
    """Calculates node positions by averaging the lens values of points in each node."""
    positions = {}
    for node_id, indices in graph_dict["nodes"].items():
        points = lens_2d[np.array(indices)]
        positions[node_id] = points.mean(axis=0)
    return positions

def mapper_to_networkx(graph_dict):
    """Converts a KeplerMapper graph dictionary into a NetworkX Graph object."""
    G = nx.Graph()
    for node_id, members in graph_dict["nodes"].items():
        G.add_node(node_id, size=len(members))
    
    links = graph_dict.get("links", [])
    if isinstance(links, dict):
        for a, neighbors in links.items():
            for b in neighbors:
                G.add_edge(a, b)
    else:
        for link in links:
            if isinstance(link, (list, tuple)) and len(link) >= 2:
                G.add_edge(link[0], link[1])
    return G

def draw_mapper_graph(graph_dict, positions, node_values=None, title="", output_file="output.png"):
    """Generates and saves a professional PNG visualization of the Mapper graph."""
    G = mapper_to_networkx(graph_dict)
    # Node size based on number of points (log scale for better visibility)
    sizes = [max(10, np.log10(G.nodes[n]["size"] + 1) * 60) for n in G.nodes()]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    nx.draw_networkx_edges(G, positions, ax=ax, width=1.2, edge_color="#666666", alpha=0.4)

    if node_values is None:
        nx.draw_networkx_nodes(G, positions, ax=ax, node_size=sizes, node_color="#3182bd")
    else:
        values = np.array([node_values.get(n, np.nan) for n in G.nodes()])
        # Handle NaNs
        finitos = values[np.isfinite(values)]
        if finitos.size == 0:
            values[:] = 0.0
        else:
            values = np.nan_to_num(values, nan=np.nanmedian(finitos))

        nodes = nx.draw_networkx_nodes(
            G, positions, ax=ax,
            node_size=sizes, node_color=values,
            cmap="viridis"
        )
        plt.colorbar(nodes, ax=ax, shrink=0.8, label="Mean value per node")

    ax.set_title(title, fontsize=20)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_file, dpi=300)
    plt.close(fig)

def get_node_means(graph_dict, data_vector):
    """Calculates the mean of a specific feature for each node in the graph."""
    return {node_id: float(np.mean(data_vector[np.array(indices)]))
            for node_id, indices in graph_dict["nodes"].items()}

def calculate_knn_eccentricity(X_emb, k=50):
    """
    Calculates the eccentricity of points based on kNN distances.
    High eccentricity indicates points that are far from the data center.
    """
    neighbors = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=-1).fit(X_emb)
    distances, _ = neighbors.kneighbors(X_emb, return_distance=True)
    # Exclude the distance to the point itself (first column)
    knn_distances = distances[:, 1:]
    return knn_distances.mean(axis=1)

def get_node_stats(graph_dict, vector):
    """Calculates the mean value of a vector for each node in the Mapper graph."""
    return {
        node_id: float(np.mean(vector[np.array(indices)]))
        for node_id, indices in graph_dict["nodes"].items()
    }