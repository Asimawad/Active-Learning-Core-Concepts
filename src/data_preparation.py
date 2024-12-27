import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def create_al_synthetic_dataset(
    n_samples=10000,
    noise=0.3,
    random_state=42,
    n_samples_test=1000,
    labeled_samples=5
):
    X, y = fetch_covtype(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=n_samples, test_size=n_samples_test, random_state=random_state, stratify=y
    )

    train_idx = np.arange(len(y_train))
    class_indices = [train_idx[y_train == c] for c in np.unique(y_train)]

    np.random.seed(random_state + 3)
    labeled_idx = np.concatenate([
        np.random.choice(class_idx, labeled_samples, replace=False)
        for class_idx in class_indices
    ])
    unlabeled_idx = np.setdiff1d(train_idx, labeled_idx)

    return X_train, y_train, X_test, y_test, labeled_idx, unlabeled_idx

def visualize_dataset(
    X_train, y_train, labeled_idx, X_test=None, y_test=None, title="Forest Covertype Dataset"
):
    plt.figure(figsize=(12, 8))
    pca = PCA(n_components=2)
    X_train_2d = pca.fit_transform(X_train)

    if X_test is not None:
        X_test_2d = pca.transform(X_test)
        plt.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c="gray", label="Test Data", alpha=0.1, zorder=-10)

    classes = np.unique(y_train)
    colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))

    for i, (c, color) in enumerate(zip(classes, colors)):
        mask = y_train == c
        plt.scatter(X_train_2d[mask][:, 0], X_train_2d[mask][:, 1], c=[color], label=f"Class {c} (Unlabeled)", alpha=0.2)

    for i, (c, color) in enumerate(zip(classes, colors)):
        mask = y_train[labeled_idx] == c
        if np.any(mask):
            plt.scatter(
                X_train_2d[labeled_idx][mask][:, 0],
                X_train_2d[labeled_idx][mask][:, 1],
                c=[color],
                label=f"Class {c} (Labeled)",
                s=100,
                edgecolor="black",
            )

    plt.title(title)
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
