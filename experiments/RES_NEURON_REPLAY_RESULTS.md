# .res reusable-neuron replay — observed results

Dataset: scikit-learn handwritten digits, stratified 75/25 split (random_state=42); 450 held-out test examples.

Original model: MLPClassifier, 64 inputs → 32 ReLU hidden units → 10 outputs, max_iter=350, random_state=42.

The trained MLP's 2,368 weights and 42 biases were saved exactly in a .res file using NumPy's NPZ binary container, and reloaded without changes. A **single reusable scalar neuron** loaded each original neuron's incoming weights and bias in turn, computed its output, and used retained intermediate layer outputs to process the next layer.

## Observed run

| Measurement | Result |
| --- | ---: |
| MLP test accuracy | 97.56% (439/450) |
| Reusable-neuron test accuracy | 97.56% (439/450) |
| Prediction agreement | 100% (450/450) |
| MLP prediction time | 0.000567279 s |
| Sequential prediction time | 0.119117451 s |
| Measured slowdown | ~210× |
| .res weight round-trip | Exact |

Timings are machine/run-specific and should not be treated as a universal performance ratio.

## Interpretation and limitation

This demonstrates **sequential execution of an existing MLP** with one reusable computational neuron, retaining the original layer topology, activation behavior, biases, and intermediate outputs. It does **not** demonstrate that a standalone one-neuron classifier, without those intermediate computations, can reproduce the MLP's accuracy. The .res format here is a simple weight-storage container, not a cross-architecture transformation algorithm.

## Reproduce

Run `python experiments/res_neuron_replay.py` with `numpy` and `scikit-learn` installed. The script writes `replay_weights.res` and `res_neuron_replay_results.json` beside itself. The script re-trains the MLP, so execution times may differ across runs.

This report records one prior local run; no independent reproduction is claimed in this commit.
