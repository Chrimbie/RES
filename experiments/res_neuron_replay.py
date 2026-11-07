"""Reproduce .res weight replay: MLP vs one reusable computational neuron.
Requires: numpy, scikit-learn
Run: python experiments/res_neuron_replay.py
"""
from pathlib import Path
import json
import time
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parent
X, y = load_digits(return_X_y=True)
X = X / 16.0
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=350,
                    random_state=42, early_stopping=False)
mlp.fit(Xtr, ytr)

start = time.perf_counter()
reference = mlp.predict(Xte)
mlp_seconds = time.perf_counter() - start

res_path = ROOT / "replay_weights.res"
with res_path.open("wb") as f:
    np.savez(f, **{f"W{i}": w for i, w in enumerate(mlp.coefs_)},
             **{f"b{i}": b for i, b in enumerate(mlp.intercepts_)})
with np.load(res_path) as data:
    weights = [data[f"W{i}"].copy() for i in range(len(mlp.coefs_))]
    biases = [data[f"b{i}"].copy() for i in range(len(mlp.intercepts_))]
assert all(np.array_equal(a, b) for a, b in zip(weights, mlp.coefs_))
assert all(np.array_equal(a, b) for a, b in zip(biases, mlp.intercepts_))

class ReusableNeuron:
    def __init__(self):
        self.weights = None
        self.bias = 0.0

    def load(self, incoming_weights, bias):
        self.weights = incoming_weights
        self.bias = float(bias)

    def compute(self, inputs, activation):
        total = self.bias
        for x, w in zip(inputs, self.weights):
            total += float(x) * float(w)
        return max(0.0, total) if activation == "relu" else total

neuron = ReusableNeuron()

def infer(sample):
    current = sample
    for layer, (W, b) in enumerate(zip(weights, biases)):
        output = np.empty(W.shape[1], dtype=np.float64)
        for j in range(W.shape[1]):
            neuron.load(W[:, j], b[j])
            output[j] = neuron.compute(
                current, "relu" if layer < len(weights) - 1 else "identity"
            )
        current = output
    return int(mlp.classes_[np.argmax(current)])

start = time.perf_counter()
replayed = np.array([infer(row) for row in Xte])
replay_seconds = time.perf_counter() - start
result = {
    "dataset": "sklearn digits",
    "test_samples": len(yte),
    "architecture": [64, 32, 10],
    "mlp_accuracy": float(accuracy_score(yte, reference)),
    "single_reusable_neuron_accuracy": float(accuracy_score(yte, replayed)),
    "prediction_agreement": float(np.mean(reference == replayed)),
    "mlp_prediction_seconds": mlp_seconds,
    "sequential_prediction_seconds": replay_seconds,
    "time_ratio": replay_seconds / mlp_seconds if mlp_seconds else None,
    "weight_count": int(sum(w.size for w in weights)),
    "bias_count": int(sum(b.size for b in biases)),
    "res_exact_roundtrip": True,
    "note": "One reused neuron processes each original MLP neuron in sequence; intermediate layer outputs are retained. This is sequential MLP execution, not a standalone one-neuron classifier."
}
(ROOT / "res_neuron_replay_results.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
