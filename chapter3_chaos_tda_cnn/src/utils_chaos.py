import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random
import hashlib
from ripser import ripser
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# REPRODUCIBILITY 

def set_seeds(seed: int = 42):
    """Sets seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# DATA GENERATION & DYNAMICS 

def logistic_map(x0, r, length=1000, transient=1000):
    """Simulates the Logistic Map: x_{n+1} = r * x_n * (1 - x_n"""
    n = length + transient
    x = np.zeros(n)
    x[0] = x0
    for i in range(1, n):
        x[i] = r * x[i-1] * (1 - x[i-1])
    return x[transient:]

def lyapunov_exponent(series, r, start_index=0, eps=1e-12):
    """Calculates the Lyapunov Exponent for the Logistic Map"""
    n = len(series)
    if n < start_index + 500:
        return 0.0
    derivative = r * (1 - 2 * series[start_index:])
    abs_der = np.abs(derivative)
    vals = np.where(abs_der > eps, abs_der, eps)
    return np.mean(np.log(vals))

def load_lorenz_data(path, le_threshold=0.01):
    """Loads Lorenz system dataset and assigns labels based on Lyapunov exponent"""
    dataset = []
    with open(path, "r") as f:
        for line in f:
            values = list(map(float, line.strip().split()))
            if len(values) != 7 + 3000:
                continue
            
            lyap = values[6]
            series_raw = values[7:]
            series = [series_raw[i:i+3] for i in range(0, len(series_raw), 3)]
            
            dataset.append({
                "x0": values[0], "y0": values[1], "z0": values[2],
                "sigma": values[3], "rho": values[4], "beta": values[5],
                "lyapunov": lyap,
                "label": 1 if lyap > le_threshold else 0,
                "series": series
            })
    return dataset

# PREPROCESSING & TDA 

def embed_takens_manual(series, dimension=3, delay=1):
    """Implements Takens' Delay Embedding Theorem"""
    embedded = []
    for i in range(len(series) - (dimension - 1) * delay):
        point = [series[i + j * delay] for j in range(dimension)]
        embedded.append(point)
    return np.array(embedded)

def hash_series(series, decimals=4):
    """Generates an MD5 hash for a series to identify duplicates."""
    q = np.round(series, decimals=decimals)
    return hashlib.md5(q.tobytes()).hexdigest()

def diagrams_to_dict(diagram):
    """Converts a list of diagrams into a dictionary {dim: diagram}."""
    if isinstance(diagram, list):
        return {i: diagram[i] for i in range(len(diagram))}
    return dict(diagram)

def birth_persistence(points):
    """Converts (birth, death) pairs to (birth, persistence)"""
    if points is None or len(points) == 0:
        return None
    # Filter infinite deaths and NaN
    pts = points[np.isfinite(points[:, 1])]
    if len(pts) == 0:
        return None
    birth = pts[:, 0]
    persistence = np.maximum(pts[:, 1] - pts[:, 0], 0.0)
    return np.column_stack([birth, persistence])

def compute_persistence(dataset, batch_size=300, homology_dims=(0, 1)):
    """Computes persistence diagrams using Ripser in batches"""
    results = []
    total_batches = (len(dataset) - 1) // batch_size + 1
    for num_batch in range(total_batches):
        start = num_batch * batch_size
        end = start + batch_size
        batch = dataset[start:end]
        for item in batch:
            X = item['embedded']
            if len(X) > 1000:
                X = X[::2]
            X = MinMaxScaler().fit_transform(X)
            dgms = ripser(X, maxdim=max(homology_dims))['dgms']
            item['diagram'] = dgms
        results.extend(batch)
    return results

def get_global_ranges(dataset, dims=(0, 1), margin=0.05):
    """Calculates global birth and persistence ranges for normalization"""
    all_pts = []
    for item in dataset:
        dct = diagrams_to_dict(item['diagram'])
        for dim in dims:
            bp = birth_persistence(dct.get(dim))
            if bp is not None and len(bp) > 0:
                all_pts.append(bp)
    if not all_pts:
        return (0.0, 1.0), (0.0, 1.0)
    all_pts = np.vstack(all_pts)
    bmin, pmin = all_pts.min(axis=0)
    bmax, pmax = all_pts.max(axis=0)
    mb = margin * max(bmax - bmin, 1e-9)
    mp = margin * max(pmax - pmin, 1e-9)
    return (bmin - mb, bmax + mb), (pmin - mp, pmax + mp)

def vectorize_persistence_images(dataset, dims=(0, 1), resolution=32, channels_per_dim=True, 
                                sigma_px=1.0, normalize=False, birth_range=None, persistence_range=None):
    """Generates Persistence Images (PI) by summing Gaussians"""
    if birth_range is None or persistence_range is None:
        b_range, p_range = get_global_ranges(dataset, dims=dims)
    else:
        b_range, p_range = birth_range, persistence_range

    bmin, bmax = b_range
    pmin, pmax = p_range
    ys, xs = np.mgrid[0:resolution, 0:resolution]
    two_sigma2 = 2.0 * (sigma_px ** 2)

    def add_gaussians(bp_points):
        if bp_points is None or len(bp_points) == 0:
            return np.zeros((resolution, resolution), dtype=np.float32)
        bx = (bp_points[:, 0] - bmin) / (bmax - bmin + 1e-12) * (resolution - 1)
        py = (bp_points[:, 1] - pmin) / (pmax - pmin + 1e-12) * (resolution - 1)
        image = np.zeros((resolution, resolution), dtype=np.float64)
        for x0, y0 in zip(bx, py):
            d2 = (xs - x0) ** 2 + (ys - y0) ** 2
            image += np.exp(-d2 / max(two_sigma2, 1e-12))
        return image.astype(np.float32)

    for item in dataset:
        dct = diagrams_to_dict(item['diagram'])
        if channels_per_dim:
            channels = []
            for dim in dims:
                bp = birth_persistence(dct.get(dim))
                img = add_gaussians(bp)
                if normalize and img.sum() > 0:
                    img /= img.sum()
                channels.append(img)
            item['persimg'] = np.stack(channels, axis=-1)
        else:
            all_bp = [birth_persistence(dct.get(dim)) for dim in dims]
            valid_bp = np.vstack([bp for bp in all_bp if bp is not None]) if any(bp is not None for bp in all_bp) else np.empty((0,2))
            img = add_gaussians(valid_bp)
            if normalize and img.sum() > 0:
                img /= img.sum()
            item['persimg'] = img[..., None]
    return dataset

# CNN TRAINING & EVALUATION 

def prepare_data_loader(X, y, batch_size=32, shuffle=False):
    """Prepares PyTorch DataLoader from numpy arrays"""
    # (N, H, W, C) -> (N, C, H, W)
    X_t = torch.tensor(X, dtype=torch.float32).permute(0, 3, 1, 2)
    y_t = torch.tensor(y, dtype=torch.long)
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

def train_cnn(model, train_loader, val_loader, epochs=2000, lr=8e-3, 
              device="cpu", patience=100, weight_decay=1e-5, class_weights=None):
    """Standard CNN training loop with early stopping"""
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(weight=(class_weights.to(device) if class_weights is not None else None))

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(Xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * Xb.size(0)
            correct += (outputs.argmax(1) == yb).sum().item()
            total += yb.size(0)

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                outputs = model(Xb)
                loss = criterion(outputs, yb)
                val_loss += loss.item() * Xb.size(0)
                val_correct += (outputs.argmax(1) == yb).sum().item()
                val_total += yb.size(0)

        history["train_loss"].append(train_loss/total)
        history["val_loss"].append(val_loss/val_total)
        history["train_acc"].append(correct/total)
        history["val_acc"].append(val_correct/val_total)

        # Early stopping logic
        if val_loss/val_total < best_val_loss - 1e-6:
            best_val_loss = val_loss/val_total
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)
    return model, history

def evaluate_model(model, loader, device="cpu"):
    """Evaluates the model and returns true labels, predictions and probabilities"""
    model.eval()
    y_true, y_pred, y_probs = [], [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            outputs = model(Xb)
            probs = torch.softmax(outputs, dim=1)
            y_true.extend(yb.numpy())
            y_pred.extend(probs.argmax(dim=1).cpu().numpy())
            y_probs.extend(probs[:, 1].cpu().numpy())
    return np.array(y_true), np.array(y_pred), np.array(y_probs)

# SPECIFIC FUNCTIONS FOR LOGISTIC MAP EXPERIMENT 

def generate_and_filter_logistic_dataset(x0_vals, r_vals, length=1000, transient=1000, threshold=0.1):
    """Generates logistic map data, labels via LE, and removes duplicates via hashing"""
    dataset = []
    for r in r_vals:
        for x0 in x0_vals:
            # We use the functions already defined in this utils file
            series = logistic_map(x0, r, length=length, transient=transient)
            if len(series) < length: continue
            le = lyapunov_exponent(series, r)
            label = 1 if le > threshold else 0
            dataset.append({'serie': series, 'label': label, 'r': r, 'x0': x0, 'le': le})
    
    seen = set()
    filtered = []
    for item in dataset:
        h = hash_series(item['serie'])
        if h not in seen:
            seen.add(h)
            filtered.append(item)
    return filtered

def balance_logistic_dataset(dataset, n_per_class=None, seed=42):
    """Balances classes specifically for the logistic dataset"""
    rng = np.random.default_rng(seed)
    cls0 = [d for d in dataset if d['label'] == 0]
    cls1 = [d for d in dataset if d['label'] == 1]
    k = min(len(cls0), len(cls1)) if n_per_class is None else n_per_class
    
    sel0 = rng.choice(cls0, size=k, replace=False).tolist()
    sel1 = rng.choice(cls1, size=k, replace=False).tolist()
    combined = sel0 + sel1
    rng.shuffle(combined)
    return combined

def compare_to_article_results(y_true, y_pred):
    """Visualization: Benchmarking against the reference article"""
    paper_results = {'test_accuracy': 99.41, 'nonchaotic_acc': 99.71, 'chaotic_acc': 99.11}
    our_acc = np.mean(y_pred == y_true) * 100
    
    print(f"\n[COMPARISON] Our TDA+CNN: {our_acc:.2f}% | Paper CNN: {paper_results['test_accuracy']}%")
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x=['TDA+CNN (Ours)', 'CNN (Article)'], y=[our_acc, paper_results['test_accuracy']])
    plt.ylim(90, 100)
    plt.title("Benchmarking against State-of-the-Art")
    plt.ylabel("Accuracy (%)")
    plt.show()

def plot_lyapunov_precision(test_ds, y_pred, y_true):
    """Visualization: Accuracy bins based on the Lyapunov Exponent"""
    le_vals = np.array([item['le'] for item in test_ds])
    bins = np.linspace(le_vals.min(), le_vals.max(), 8)
    indices = np.digitize(le_vals, bins)
    
    accuracies = []
    for i in range(1, len(bins)):
        mask = (indices == i)
        if np.sum(mask) > 0:
            accuracies.append(np.mean(y_pred[mask] == y_true[mask]) * 100)
        else: accuracies.append(0)

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(accuracies)), accuracies, color='orange')
    plt.axhline(y=90, color='r', linestyle='--', label='90% Threshold')
    plt.xlabel('Lyapunov Exponent Range')
    plt.ylabel('Accuracy (%)')
    plt.xticks(range(len(accuracies)), [f"{bins[i]:.2f}" for i in range(len(bins)-1)])
    plt.legend()
    plt.show()

# SPECIFIC FUNCTIONS FOR LORENZ 3D EXPERIMENT

def embed_direct_lorenz(dataset, keep_last=None, subsampling=1, normalize=True):
    """Prepares point clouds directly from the (x, y, z) series"""
    for item in dataset:
        S = np.asarray(item['serie'], dtype=np.float64)  # (T, 3)
        if keep_last is not None and keep_last > 0 and len(S) > keep_last:
            S = S[-keep_last:]
        if subsampling and subsampling > 1:
            S = S[::subsampling]
        if normalize:
            S = MinMaxScaler().fit_transform(S)
        item['embedded'] = S
    return dataset

def run_multi_seed_experiment_lorenz(paths, n_runs=5, le_threshold=0.01, epochs=2000):
    """Executes a multi-seed robustness analysis for the Lorenz 3D system"""
    from src.models import PersistenceCNN
    seeds = [42, 678, 3002, 43654, 81828][:n_runs]
    
    # Accumulators for final statistics
    accs_train, accs_val, accs_test = [], [], []
    losses_train, losses_val, losses_test = [], [], []
    accs_class0, accs_class1 = [], []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i, seed in enumerate(seeds):
        print(f"\nEXPERIMENT {i+1}/{n_runs} (Seed {seed})")
        set_seeds(seed)

        # 1. Load and Preprocess (Reloaded each run to ensure seed consistency)
        train_ds = embed_direct_lorenz(load_lorenz_data(paths["train"], le_threshold))
        val_ds   = embed_direct_lorenz(load_lorenz_data(paths["val"], le_threshold))
        test_ds  = embed_direct_lorenz(load_lorenz_data(paths["test"], le_threshold))

        # 2. TDA & Persistence Images
        train_ds = compute_persistence(train_ds)
        val_ds   = compute_persistence(val_ds)
        test_ds  = compute_persistence(test_ds)

        b_range, p_range = get_global_ranges(train_ds)
        vectorize_persistence_images(train_ds, birth_range=b_range, persistence_range=p_range)
        vectorize_persistence_images(val_ds, birth_range=b_range, persistence_range=p_range)
        vectorize_persistence_images(test_ds, birth_range=b_range, persistence_range=p_range)

        # 3. Training Preparation
        def get_xy(ds):
            return np.stack([it['persimg'] for it in ds]), np.array([it['label'] for it in ds])
        
        X_tr, y_tr = get_xy(train_ds)
        X_va, y_va = get_xy(val_ds)
        X_te, y_te = get_xy(test_ds)

        # Class Weights for balancing
        cls_counts = np.bincount(y_tr, minlength=2)
        cls_weights = torch.tensor((cls_counts.max() / np.clip(cls_counts, 1, None)).astype("float32"))

        train_loader = prepare_data_loader(X_tr, y_tr, batch_size=128, shuffle=True)
        val_loader   = prepare_data_loader(X_va, y_va, batch_size=128)
        test_loader  = prepare_data_loader(X_te, y_te, batch_size=128)

        # 4. CNN Training
        model = PersistenceCNN(input_channels=X_tr.shape[-1], num_classes=2)
        model, history = train_cnn(model, train_loader, val_loader, 
                                   epochs=epochs, dispositivo=device, class_weights=cls_weights)

        # 5. Evaluate and Record
        y_true, y_pred, _ = evaluate_model(model, test_loader, dispositivo=device)
        
        # Calculate loss manually for the test set
        criterion = torch.nn.CrossEntropyLoss(weight=cls_weights.to(device))
        test_loss = 0
        model.eval()
        with torch.no_grad():
            for xb, yb in test_loader:
                out = model(xb.to(device))
                test_loss += criterion(out, yb.to(device)).item() * xb.size(0)
        
        # Store results
        accs_train.append(history["train_acc"][-1])
        accs_val.append(history["val_acc"][-1])
        accs_test.append((y_pred == y_true).mean())
        losses_train.append(history["train_loss"][-1])
        losses_val.append(history["val_loss"][-1])
        losses_test.append(test_loss / len(y_true))

        # Class-specific accuracy
        cm = confusion_matrix(y_true, y_pred)
        acc_per_cls = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
        accs_class0.append(acc_per_cls[0])
        accs_class1.append(acc_per_cls[1])

        print(f"Run {i+1} Finished. Test Accuracy: {accs_test[-1]:.4f}")

    # 6. Final Global Stats
    def stats(arr): return np.mean(arr), np.std(arr)

    print("FINAL MULTI-SEED RESULTS")
    print(f"Train Acc : {stats(accs_train)[0]:.4f} ± {stats(accs_train)[1]:.4f}")
    print(f"Val   Acc : {stats(accs_val)[0]:.4f} ± {stats(accs_val)[1]:.4f}")
    print(f"Test  Acc : {stats(accs_test)[0]:.4f} ± {stats(accs_test)[1]:.4f}")
    print(f"Test Class 0 Acc: {stats(accs_class0)[0]:.4f} ± {stats(accs_class0)[1]:.4f}")
    print(f"Test Class 1 Acc: {stats(accs_class1)[0]:.4f} ± {stats(accs_class1)[1]:.4f}")
    print(f"Test Loss: {stats(losses_test)[0]:.4f} ± {stats(losses_test)[1]:.4f}")