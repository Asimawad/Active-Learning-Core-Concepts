import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.active_learning_strategies import (
    least_confident,
    margin_sampling,
    entropy_sampling,
    query_by_committee
)

def active_learning_loop(method="random", epochs=751, seed=42):
    from src.data_preparation import create_al_synthetic_dataset

    X_train, y_train, X_test, y_test, labeled_idx, unlabeled_idx = (
        create_al_synthetic_dataset(labeled_samples=5)
    )

    results = {}
    print(f"Starting {method} sampling training loop")

    for _ in range(labeled_idx.shape[0], epochs):
        model = RandomForestClassifier(random_state=seed)
        model.fit(X_train[labeled_idx], y_train[labeled_idx])

        unlabeled_pool = X_train[unlabeled_idx]

        if method == "confidence":
            selected = least_confident(model, unlabeled_pool)
        elif method == "margin":
            selected = margin_sampling(model, unlabeled_pool)
        elif method == "entropy":
            selected = entropy_sampling(model, unlabeled_pool)
        elif method == "random":
            selected = np.random.choice(len(unlabeled_idx), 1, replace=False)[0]
        elif method == "qbc":
            models = []
            for _ in range(3):
                bootstrap_idx = np.random.choice(labeled_idx, len(labeled_idx), replace=True)
                rndf_model = RandomForestClassifier()
                rndf_model.fit(X_train[bootstrap_idx], y_train[bootstrap_idx])
                models.append(rndf_model)
            selected = query_by_committee(models, unlabeled_pool)

        selected_sample_idx = unlabeled_idx[selected]
        labeled_idx = np.append(labeled_idx, selected_sample_idx)
        unlabeled_idx = np.delete(unlabeled_idx, selected)

        acc = model.score(X_test, y_test)
        results[labeled_idx.shape[0]] = acc

        if labeled_idx.shape[0] % 50 == 0:
            print(f"Training using {labeled_idx.shape[0]} labeled data points: Test Accuracy = {acc}")

    return results
