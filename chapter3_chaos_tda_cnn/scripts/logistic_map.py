import numpy as np
import torch
from src.utils_chaos import (
    set_seeds, compute_persistence, vectorize_persistence_images,
    get_global_ranges, prepare_data_loader, train_cnn, evaluate_model,
    
    generate_and_filter_logistic_dataset, balance_logistic_dataset,
    compare_to_article_results, plot_lyapunov_precision
)
from src.models import PersistenceCNN

# CONFIGURATION 
x0_TRAIN, x0_VAL, x0_TEST = [0.3, 0.9], [0.55], [0.8]
R_MESH = np.linspace(0.0, 4.0, 10000, endpoint=False)
LE_THRESHOLD = 0.1

def run_logistic_pipeline():
    print("Logistic Map Chaos Detection (TDA + CNN)")
    set_seeds(42)

    # 1. Data Generation
    train_ds = balance_logistic_dataset(generate_and_filter_logistic_dataset(x0_TRAIN, R_MESH))
    val_ds = balance_logistic_dataset(generate_and_filter_logistic_dataset(x0_VAL, R_MESH))
    test_ds = balance_logistic_dataset(generate_and_filter_logistic_dataset(x0_TEST, R_MESH))

    # 2. TDA Pipeline
    from src.utils_chaos import embed_takens_manual
    for ds in [train_ds, val_ds, test_ds]:
        for item in ds:
            item['embedded'] = embed_takens_manual(item['serie'], dimension=3, delay=1)
    
    train_ds = compute_persistence(train_ds); val_ds = compute_persistence(val_ds); test_ds = compute_persistence(test_ds)

    # 3. Persistence Images
    b_range, p_range = get_global_ranges(train_ds)
    vectorize_persistence_images(train_ds, birth_range=b_range, persistence_range=p_range)
    vectorize_persistence_images(val_ds, birth_range=b_range, persistence_range=p_range)
    vectorize_persistence_images(test_ds, birth_range=b_range, persistence_range=p_range)

    # 4. Training
    def stack_data(ds):
        return np.stack([it['persimg'] for it in ds]), np.array([it['label'] for it in ds])
    
    X_train, y_train = stack_data(train_ds)
    X_val, y_val = stack_data(val_ds)
    X_test, y_test = stack_data(test_ds)

    train_ld = prepare_data_loader(X_train, y_train, batch_size=128, shuffle=True)
    val_ld = prepare_data_loader(X_val, y_val, batch_size=128)
    test_ld = prepare_data_loader(X_test, y_test, batch_size=128)

    model = PersistenceCNN(input_channels=2, num_classes=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = train_cnn(model, train_ld, val_ld, device=device)

    # 5. Final Evaluation
    y_true, y_pred, _ = evaluate_model(model, test_ld, device=device)
    compare_to_article_results(y_true, y_pred)
    plot_lyapunov_precision(test_ds, y_pred, y_true)

    torch.save(model.state_dict(), 'models/persistence_cnn_logistic.pth')

if __name__ == "__main__":
    run_logistic_pipeline()