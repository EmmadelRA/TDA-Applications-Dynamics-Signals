import numpy as np
import pickle
import json
from pathlib import Path

# Configuration 
BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES_DIR = BASE_DIR / "data" / "processed"
REPRESENTATIVES_DIR = FEATURES_DIR / "representatives"
OUTPUT_DIR = BASE_DIR / "data" / "mapper_input"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def build_final_dataset():
    """
    Consolidates all patient representatives into a single dataset and 
    generates detailed metadata for full traceability.
    """
    print(f"Searching for representatives in: {REPRESENTATIVES_DIR}")
    repr_files = list(REPRESENTATIVES_DIR.glob("*_repr.pkl"))
    
    all_points = []
    detailed_metadata = []
    patient_mapping = {}

    for i, repr_file in enumerate(repr_files):
        try:
            patient_id = repr_file.stem.replace("_repr", "")
            
            if i % 500 == 0:
                print(f"Processing {i}/{len(repr_files)}: {patient_id}...")

            # 1. Load representatives and their local filtered indices
            with open(repr_file, "rb") as f:
                representatives = pickle.load(f)  # (k, 15)
            local_indices = np.load(REPRESENTATIVES_DIR / f"{patient_id}_repr_idx.npy")

            # 2. Load original features to map back to global indices
            orig_file = FEATURES_DIR / f"{patient_id}_features.pkl"
            with open(orig_file, "rb") as f:
                orig_data = np.array(pickle.load(f), dtype=np.float32)

            # Reconstruct the "valid windows" mask (label is in column 15)
            valid_mask = (orig_data[:, 15] == 0)
            global_indices_map = np.where(valid_mask)[0]

            # Map local representative indices back to original file indices
            original_indices = global_indices_map[local_indices]

            # 3. Store data and metadata
            for j, (point, orig_idx) in enumerate(zip(representatives, original_indices)):
                all_points.append(point)
                
                meta_entry = {
                    "patient_id": patient_id,
                    "point_idx": len(all_points) - 1,
                    "original_file_idx": int(orig_idx)
                }
                detailed_metadata.append(meta_entry)
                patient_mapping.setdefault(patient_id, []).append(int(orig_idx))

        except Exception as e:
            print(f"Error processing {patient_id}: {e}")
            continue

    # Save Outputs 
    X_mapper = np.array(all_points, dtype=np.float32)
    np.save(OUTPUT_DIR / "X_mapper.npy", X_mapper)

    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(detailed_metadata, f, indent=2)

    with open(OUTPUT_DIR / "patient_index_map.json", "w") as f:
        json.dump(patient_mapping, f, indent=2)

    print(f"\nDataset Ready:")
    print(f" - Total Points: {X_mapper.shape[0]}")
    print(f" - Features: {X_mapper.shape[1]}")
    print(f" - Saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    build_final_dataset()
