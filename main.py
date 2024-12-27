# Main script to organize the project
import numpy as np
import matplotlib.pyplot as plt
from src.data_preparation import create_al_synthetic_dataset, visualize_dataset
from src.active_learning_strategies import (
    least_confident,
    margin_sampling,
    entropy_sampling,
    query_by_committee,
    compute_vote_entropy
)
from src.active_learning_loop import active_learning_loop
from src.visualization import plot_learning_curves

# Example usage
if __name__ == "__main__":
    random_results = active_learning_loop(method='random', epochs=751)
    confidence_results = active_learning_loop(method='confidence', epochs=751)
    margin_results = active_learning_loop(method='margin', epochs=751)
    entropy_results = active_learning_loop(method='entropy', epochs=751)
    qbc_results = active_learning_loop(method='qbc', epochs=751)

    # Plot learning curves
    plot_learning_curves(random_results, confidence_results, margin_results, entropy_results, qbc_results)
