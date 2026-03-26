import numpy as np                          
import matplotlib.pyplot as plt  
from ripser import ripser
from persim import plot_diagrams
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
from src.utils_chaos import load_lorenz_data, embed_takens_manual

# CONFIGURATION 
DATA_PATH = "data/lorenz/LS_TRAIN_Data_Paper_norm.txt"
OUTPUT_DIR = Path("results/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Takens combinations to showcase (Dimension d, Delay tau)
COMBINATIONS = [(10, 2), (6, 2)]
POINTS_TO_PLOT = 500  # We use the last points for better visibility

def generate_takens_plots():
    """
    Generates visual proof of Takens' Theorem by comparing Regular vs Chaotic 
    attractors and their Persistence Diagrams
    """
    print(f"Loading data from {DATA_PATH}...")
    dataset = load_lorenz_data(DATA_PATH, le_threshold=0.01)

    # 1. Select representative samples
    serie_nc = None
    serie_c = None

    for item in dataset:
        if item['label'] == 0 and serie_nc is None:
            serie_nc = item['serie'][-POINTS_TO_PLOT:]
        elif item['label'] == 1 and serie_c is None:
            serie_c = item['serie'][-POINTS_TO_PLOT:]
        if serie_nc is not None and serie_c is not None:
            break

    # FIGURE 1: TIME SERIES 
    print("Generating Figure 1: Time Series comparison...")
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot([x[0] for x in serie_nc], lw=1, color='blue')
    plt.xlabel("Time")
    plt.ylabel("x-variable")
    plt.title("Non-Chaotic (Periodic) Series", fontsize=14)
    
    plt.subplot(1, 2, 2)
    plt.plot([x[0] for x in serie_c], lw=1, color='red')
    plt.xlabel("Time")
    plt.ylabel("x-variable")
    plt.title("Chaotic Series", fontsize=14)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "lorenz_series_comparison.png")
    plt.show()

    # FIGURE 2: PERSISTENCE DIAGRAMS (TAKENS) 
    print("Generating Figure 2: Persistence Diagrams across embedding dimensions...")
    fig, axs = plt.subplots(len(COMBINATIONS), 2, figsize=(12, 10))
    
    for i, (dim, tau) in enumerate(COMBINATIONS):
        # Process Non-Chaotic
        emb_nc = embed_takens_manual(serie_nc, dimension=dim, delay=tau)
        emb_nc = MinMaxScaler().fit_transform(emb_nc)
        dgms_nc = ripser(emb_nc, maxdim=2)['dgms']
        
        # Process Chaotic
        emb_c = embed_takens_manual(serie_c, dimension=dim, delay=tau)
        emb_c = MinMaxScaler().fit_transform(emb_c)
        dgms_c = ripser(emb_c, maxdim=2)['dgms']
        
        # Plot Non-Chaotic
        plot_diagrams(dgms_nc, show=False, ax=axs[i, 0])
        axs[i, 0].set_title(f"Non-Chaotic PD (d={dim}, τ={tau})")
        
        # Plot Chaotic
        plot_diagrams(dgms_c, show=False, ax=axs[i, 1])
        axs[i, 1].set_title(f"Chaotic PD (d={dim}, τ={tau})")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "takens_persistence_comparison.png")
    plt.show()
    
    print(f"Visualizations completed. Figures saved in {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_takens_plots()
