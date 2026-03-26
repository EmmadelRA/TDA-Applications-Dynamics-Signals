import os
import gzip
import pickle
import numpy as np
from joblib import Parallel, delayed
from pathlib import Path
from src.utils import check_window_quality, extract_features #tools from utils

# Configuration 
FS = 250
WINDOW_DURATION = 25 
WINDOW_SAMPLES = FS * WINDOW_DURATION


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = OUTPUT_DIR / "valid_windows_log.txt"

def process_patient(patient_data):
    """Processes all 25s windows for a single patient."""
    
    patient_features = []
    
    if not isinstance(patient_data, (np.ndarray, list)):
        return None

    data = np.array(patient_data)
    for start in range(0, len(data) - WINDOW_SAMPLES + 1, WINDOW_SAMPLES):
        window = data[start:start + WINDOW_SAMPLES]
        
        # Quality & Labeling
        label, window_z = check_window_quality(window, fs=FS)
        if window_z is not None:
            feats = extract_features(window_z, fs=FS)
            # Add binary label: 0 for valid, 1 for rare
            patient_features.append(np.append(feats, 0 if label == "valid" else 1))

    return np.vstack(patient_features) if patient_features else None

def run_preprocessing():
    """Main execution loop for all data files."""
    data_files = list(INPUT_DIR.glob("*_batched.pkl.gz"))
    
    with open(LOG_FILE, "w") as log:
        log.write("file_name\tvalid_windows_count\n")

    for file_path in data_files:
        print(f"Processing: {file_path.name}")
        with gzip.open(file_path, 'rb') as f:
            raw_data = pickle.load(f)

        # Parallel processing of patients within the file
        results = Parallel(n_jobs=-1)(
            delayed(process_patient)(patient) for patient in raw_data
        )

        valid_arrays = [res for res in results if res is not None]
        total_valid = sum(arr.shape[0] for arr in valid_arrays) if valid_arrays else 0

        # Save results
        if valid_arrays:
            final_features = np.vstack(valid_arrays)
            output_name = file_path.name.replace(".pkl.gz", "_features.pkl")
            with open(OUTPUT_DIR / output_name, "wb") as f:
                pickle.dump(final_features, f)
        
        with open(LOG_FILE, "a") as log:
            log.write(f"{file_path.name}\t{total_valid}\n")

if __name__ == "__main__":
    run_preprocessing()

