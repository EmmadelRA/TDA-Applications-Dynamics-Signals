import numpy as np
import torch
import torch.nn as nn
from src.utils_chaos import (
    set_seeds, load_lorenz_data, embed_direct_lorenz, compute_persistence, 
    vectorize_persistence_images, get_global_ranges, prepare_data_loader, 
    train_cnn, evaluate_model
)
from src.models import PersistenceCNN
from pathlib import Path

# CONFIGURATION 
PATHS = {
    "train": "data/lorenz/LS_TRAIN_Data_Paper_norm.txt",
    "val": "data/lorenz/LS_VALIDATION_Data_Paper_norm.txt",
    "test": "data/lorenz/LS_TEST_Data_Paper_norm.txt"
}
LE_THRESHOLD = 0.01
RESOLUTION = 32
SIGMA_PX = 1.0
BATCH_SIZE = 128
LEARNING_RATE = 8e-3
EPOCHS = 2000

def run_lorenz_3d_pipeline():
    print("Lorenz System 3D (TDA + CNN)")
    set_seeds(42)

    # 1. Load and Preprocess Data
    print("Loading datasets...")
    train_ds = embed_direct_lorenz(load_lorenz_data(PATHS["train"], le_threshold=LE_THRESHOLD))
    val_ds = embed_direct_lorenz(load_lorenz_data(PATHS["val"], le_threshold=LE_THRESHOLD))
    test_ds = embed_direct_lorenz(load_lorenz_data(PATHS["test"], le_threshold=LE_THRESHOLD))

    # 2. TDA Pipeline (H0 and H1)
    print("Computing Persistent Homology...")
    train_ds = compute_persistence(train_ds, dimensiones_homologia=(0, 1))
    val_ds = compute_persistence(val_ds, dimensiones_homologia=(0, 1))
    test_ds = compute_persistence(test_ds, dimensiones_homologia=(0, 1))

    # 3. Persistence Images (Global Range Normalization)
    print("Vectorizing into Persistence Images...")
    b_range, p_range = get_global_ranges(train_ds, dims=(0, 1))
    
    params_pi = {
        "dims": (0, 1), "resolucion": RESOLUTION, "sigma_px": SIGMA_PX,
        "birth_range": b_range, "persistence_range": p_range
    }
    vectorize_persistence_images(train_ds, **params_pi)
    vectorize_persistence_images(val_ds, **params_pi)
    vectorize_persistence_images(test_ds, **params_pi)

    # 4. Neural Network Training
    def get_xy(ds):
        return np.stack([it['persimg'] for it in ds]), np.array([it['label'] for it in ds])

    X_train, y_train = get_xy(train_ds)
    X_val, y_val = get_xy(val_ds)
    X_test, y_test = get_xy(test_ds)

    train_ld = prepare_data_loader(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True)
    val_ld = prepare_data_loader(X_val, y_val, batch_size=BATCH_SIZE)
    test_ld = prepare_data_loader(X_test, y_test, batch_size=BATCH_SIZE)

    # Class weights for balancing
    counts = np.bincount(y_train)
    weights = torch.tensor((counts.max() / counts).astype(np.float32))

    model = PersistenceCNN(input_channels=2, num_classes=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Training on {device}...")
    model, history = train_cnn(model, train_ld, val_ld, epochs=EPOCHS, 
                               lr=LEARNING_RATE, dispositivo=device, class_weights=weights)

    # 5. Final Evaluation
    print("\n[FINAL RESULTS]")
    y_true, y_pred, _ = evaluate_model(model, test_ld, dispositivo=device)
    
    from sklearn.metrics import classification_report
    print(classification_report(y_true, y_pred, target_names=["Non-Chaotic", "Chaotic"]))

    # Save
    save_path = Path("models/persistence_cnn_lorenz_3d.pth")
    save_path.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    run_lorenz_3d_pipeline()

