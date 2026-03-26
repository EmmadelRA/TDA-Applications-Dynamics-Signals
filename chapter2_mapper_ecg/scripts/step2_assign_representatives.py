import pandas as pd
import numpy as np
from pathlib import Path
from src.utils import adjust_k_distribution

# Configuration 
TARGET_D = 100000   # Total points for Mapper input 
MIN_K = 3           # Minimum representatives per patient [cite: 1189]
MAX_K = 50          # Maximum representatives per patient [cite: 1189]

# Path handling
BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES_DIR = BASE_DIR / "data" / "processed"
LOG_FILE = FEATURES_DIR / "valid_windows_log.txt"
OUTPUT_CSV = FEATURES_DIR / "k_assignment.csv"

def run_assignment():
    """
    Assigns the number of representatives (k) per patient proportional 
    to their available valid windows. [cite: 1189, 1191]
    """
    if not LOG_FILE.exists():
        print(f"Error: {LOG_FILE} not found. Run step 1 first.")
        return

    # Load processing log from step 1
    df = pd.read_csv(LOG_FILE, sep="\t")
    df.columns = ["file_name", "valid_windows"]

    total_windows = df["valid_windows"].sum()

    # 1. Proportional assignment [cite: 1189]
    df["k_prop"] = (df["valid_windows"] * TARGET_D / total_windows).round().astype(int)
    
    # 2. Initial clipping [cite: 1189]
    df["k"] = df["k_prop"].clip(lower=MIN_K, upper=MAX_K)

    # 3. Final adjustment to match TARGET_D exactly 
    print(f"Initial sum: {df['k'].sum()}")
    df = adjust_k_distribution(df, TARGET_D, MIN_K, MAX_K)
    
    # Save results
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Final sum adjusted to: {df['k'].sum()}")
    print(f"Assignment saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    run_assignment()

