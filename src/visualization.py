import matplotlib.pyplot as plt

def plot_learning_curves(random_results, confidence_results, margin_results, entropy_results, qbc_results):
    plt.figure(figsize=(15, 10))
    plt.plot(list(random_results.keys()), list(random_results.values()), label="Random Sampling", c="r")
    plt.plot(list(confidence_results.keys()), list(confidence_results.values()), label="Confidence Sampling", c="g")
    plt.plot(list(margin_results.keys()), list(margin_results.values()), label="Margin Sampling", c="b")
    plt.plot(list(entropy_results.keys()), list(entropy_results.values()), label="Entropy Sampling", c="y")
    plt.plot(list(qbc_results.keys()), list(qbc_results.values()), label="QBC Sampling", c="teal")
    plt.xlabel("Number of labeled examples")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.title("Active Learning Training Accuracy Comparison")
    plt.tight_layout()
    plt.show()
