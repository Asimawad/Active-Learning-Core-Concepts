import numpy as np

def compute_vote_entropy(predictions):
    predictions = np.array(predictions)
    vote_counts = np.apply_along_axis(lambda x: np.bincount(x, minlength=np.max(predictions)+1), axis=0, arr=predictions)
    vote_probs = vote_counts / len(predictions)
    vote_entropy = -np.sum(vote_probs * np.log(vote_probs + 1e-10), axis=0)
    return vote_entropy

def least_confident(model, unlabeled_pool):
    probs = model.predict_proba(unlabeled_pool)
    confidence = np.max(probs, axis=1)
    return np.argmin(confidence)

def margin_sampling(model, unlabeled_pool):
    probs = model.predict_proba(unlabeled_pool)
    sorted_probs = np.sort(probs, axis=1)
    margins = sorted_probs[:, -1] - sorted_probs[:, -2]
    return np.argmin(margins)

def entropy_sampling(model, unlabeled_pool):
    probs = model.predict_proba(unlabeled_pool)
    entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
    return np.argmax(entropy)

def query_by_committee(models, unlabeled_pool):
    predictions = [model.predict(unlabeled_pool) for model in models]
    disagreement = compute_vote_entropy(predictions)
    return np.argmax(disagreement)
