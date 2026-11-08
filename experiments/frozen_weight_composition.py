"""Reproducible frozen-weight composition experiment.

Run: python experiments/frozen_weight_composition.py
No task router, no training of the composed network, and both branches execute.
"""
import json
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

SEED = 42
rng = np.random.default_rng(SEED)
X = rng.uniform(-1, 1, (4000, 2))
y_a = (X[:, 0] > 0).astype(int)
y_b = (X[:, 1] > 0).astype(int)
train, test = X[:3000], X[3000:]
a = MLPClassifier(hidden_layer_sizes=(8,), random_state=1, max_iter=500).fit(train, y_a[:3000])
b = MLPClassifier(hidden_layer_sizes=(8,), random_state=2, max_iter=500).fit(train, y_b[:3000])

# Copy frozen weights into a *single* block-sparse network with two outputs.
# Both sets of hidden neurons execute for every input. No selector or training.
W1 = np.concatenate((a.coefs_[0], b.coefs_[0]), axis=1)
b1 = np.concatenate((a.intercepts_[0], b.intercepts_[0]))
W2 = np.zeros((16, 2))
W2[:8, 0] = a.coefs_[1][:, 0]
W2[8:, 1] = b.coefs_[1][:, 0]
b2 = np.array([a.intercepts_[1][0], b.intercepts_[1][0]])
def composed_predict(x):
    hidden = np.maximum(0, x @ W1 + b1)
    logits = hidden @ W2 + b2
    return (logits > 0).astype(int)

pred = composed_predict(test)
truth = np.column_stack((y_a[3000:], y_b[3000:]))
original = np.column_stack((a.predict(test), b.predict(test)))
results = {
    "seed": SEED, "train_rows": 3000, "test_rows": 1000,
    "network_a_accuracy": float(accuracy_score(truth[:, 0], original[:, 0])),
    "network_b_accuracy": float(accuracy_score(truth[:, 1], original[:, 1])),
    "composed_a_accuracy": float(accuracy_score(truth[:, 0], pred[:, 0])),
    "composed_b_accuracy": float(accuracy_score(truth[:, 1], pred[:, 1])),
    "composed_both_correct": float(np.mean(np.all(pred == truth, axis=1))),
    "exact_agreement_with_originals": bool(np.array_equal(pred, original)),
}
print(json.dumps(results, indent=2))
assert results["exact_agreement_with_originals"]
