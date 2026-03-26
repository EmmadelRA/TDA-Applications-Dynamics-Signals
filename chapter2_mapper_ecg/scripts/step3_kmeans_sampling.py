import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from src.utils import get_reproducible_seed, find_medoids

# Configuration
BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES_DIR = BASE_DIR / "data" / "processed"
ASSIGNMENT_FILE = FEATURES_DIR / "k_assignment.csv" # from step 2
OUTPUT_DIR = FEATURES_DIR / "representatives"
OUTPUT_DIR.mkdir(exist_ok=True)

def run_kmeans_sampling():
    """
    Applies KMeans clustering to select 'k' representative windows per patient.
    """
    if not ASSIGNMENT_FILE.exists():
        print(f"Error: {ASSIGNMENT_FILE} not found. Run step 2 first")
        return

    assignments = pd.read_csv(ASSIGNMENT_FILE)
    print(f"Processing {len(assignments)} patients...")

    for i, row in assignments.iterrows():
        patient_id = row['file_name'].replace("_features.pkl", "")
        k = int(row['k'])
        
        feature_path = FEATURES_DIR / f"{patient_id}_features.pkl"
        if not feature_path.exists(): continue

        with open(feature_path, "rb") as f:
            # (n, 16): 15 features + 1 quality label
            data = np.array(pickle.load(f), dtype=np.float32)

        X_feats = data[:, :15]
        y_quality = data[:, 15].astype(np.int32)

        # we work just with label 0 windows (valids)
        valid_mask = (y_quality == 0)
        X_valid = X_feats[valid_mask]
        
        n_samples = X_valid.shape[0]
        seed = get_reproducible_seed(patient_id)

        if n_samples == 0:
            print(f"Skipping {patient_id}: No valid windows")
            continue

        if n_samples <= k:
            #  fewer data points than requested
            representatives, indices = X_valid, np.arange(n_samples)
        else:
            # Standarization
            scaler = StandardScaler()
            X_std = scaler.fit_transform(X_valid)

            # KMeans 
            model = KMeans(n_clusters=k, random_state=seed, n_init="auto")
            cluster_labels = model.fit_predict(X_std)
            
            # medoids (real points)
            indices = find_medoids(X_std, model.cluster_centers_, cluster_labels, k)
            representatives = X_valid[indices]

        # Guardar resultados
        output_base = OUTPUT_DIR / patient_id
        with open(f"{output_base}_repr.pkl", "wb") as f:
            pickle.dump(representatives, f)
        np.save(f"{output_base}_repr_idx.npy", indices)

    print("K-Means sampling completed successfully")

if __name__ == "__main__":
    run_kmeans_sampling()
