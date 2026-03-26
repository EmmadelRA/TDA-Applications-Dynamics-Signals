import numpy as np
from scipy.stats import skew
from scipy.signal import find_peaks
import zlib
from scipy.spatial.distance import cdist

def check_window_quality(ecg_window, fs=250, std_min=0.02, max_z=8, prominence=0.3):
    """
    Validates the quality of an ECG window based on signal variance and artifacts.
    """
    if np.std(ecg_window) < std_min:
        return None, None
    
    # Z-score normalization
    window_z = (ecg_window - np.mean(ecg_window)) / (np.std(ecg_window) + 1e-8)
    
    # Artifact rejection
    if np.max(window_z) > max_z or np.min(window_z) < -max_z:
        return None, None
    
    # R-peak detection for coarse validity check
    peaks, _ = find_peaks(window_z, distance=int(0.3*fs), prominence=prominence)
    label = "valid" if len(peaks) >= 2 else "rare"
    
    return label, window_z

def extract_features(ecg_window, fs=250):
    """
    Extracts 15 physiological features (Time, Frequency, HRV, and Morphology).
    """
    features = []
    
    # 1. Time-domain statistics (Split into 3 sub-segments)
    subsegments = np.array_split(ecg_window, 3)
    for seg in subsegments:
        features.append(np.mean(seg))
        features.append(np.std(seg))
        features.append(skew(seg))
    
    # 2. Frequency-domain features (FFT)
    fft_vals = np.abs(np.fft.fft(ecg_window))
    freqs = np.fft.fftfreq(len(ecg_window), 1/fs)
    lf = np.sum(fft_vals[(freqs >= 0.04) & (freqs < 0.15)])
    hf = np.sum(fft_vals[(freqs >= 0.15) & (freqs < 0.4)])
    features.extend([lf, hf])
    
    # 3. Heart Rate Variability (HRV) - RR Intervals
    peaks, _ = find_peaks(ecg_window, distance=0.6*fs)
    rr_intervals = np.diff(peaks)/fs
    if len(rr_intervals) > 0:
        features.append(np.mean(rr_intervals))
        features.append(np.std(rr_intervals))
    else:
        features.extend([0, 0])
    
    # 4. Morphological features
    features.append(np.max(ecg_window))
    features.append(np.min(ecg_window))
    
    return np.array(features)

def adjust_k_distribution(df, target_total, min_k, max_k):
    """
    Adjusts the proportional distribution of k to ensure the total sum 
    matches target_total exactly, respecting min/max constraints. [cite: 1189, 1191]
    """
    current_sum = df["k"].sum()
    diff = int(target_total - current_sum)

    if diff != 0:
        # Prioritize adjustments on patients with more data [cite: 1189]
        idx_order = df.sort_values("valid_windows", ascending=False).index
        i = 0
        step = 1 if diff > 0 else -1
        
        while diff != 0:
            idx = idx_order[i]
            new_val = df.at[idx, "k"] + step
            
            if min_k <= new_val <= max_k:
                df.at[idx, "k"] = new_val
                diff -= step
            
            i = (i + 1) % len(idx_order) # Loop back if needed
            
    return df


def get_reproducible_seed(name):
    """Generates a stable seed for algorithms based on patient string ID."""
    return zlib.adler32(name.encode("utf-8")) & 0xFFFFFFFF

def find_medoids(features_std, centroids, labels, k):
    """
    Finds the actual data points (medoids) closest to KMeans centroids.
    This ensures each representative is an existing signal window.
    """
    # Calculate squared euclidean distance from points to centroids
    distances = cdist(features_std, centroids, metric="sqeuclidean")
    medoid_indices = []
    
    for j in range(k):
        # Indices of points belonging to cluster j
        cluster_mask = np.where(labels == j)[0]
        if cluster_mask.size > 0:
            # Pick the point in the cluster with the minimum distance to its centroid
            best_local_idx = cluster_mask[np.argmin(distances[cluster_mask, j])]
            medoid_indices.append(best_local_idx)
        else:
            # Fallback for empty clusters
            medoid_indices.append(np.random.randint(0, len(features_std)))
            
    return np.array(medoid_indices, dtype=np.int64)
