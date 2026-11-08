# Frozen-weight composition experiment (2026-11-08)

Reproduction: `python experiments/frozen_weight_composition.py` (NumPy and scikit-learn from repository requirements).

Two independent 2→8→1 ReLU MLP classifiers were trained on 3,000 synthetic 2D inputs: A predicts whether x[0]>0; B predicts whether x[1]>0. Both were evaluated on the same 1,000 held-out samples. Their learned weights and biases were copied into a single 2→16→2 block-sparse network. The composed network was **not trained**. Both groups of eight hidden units execute on every input; there is **no input-dependent routing**.

Observed results (seed 42):
- Original A accuracy: 0.997
- Original B accuracy: 0.995
- Composed output A accuracy: 0.997
- Composed output B accuracy: 0.995
- Both outputs correct on the same sample: 0.992
- Composed predictions exactly matched both original networks: true

**Interpretation:** A single untrained composite can reproduce two independently trained networks through exact structural weight reuse. This is a block-sparse composition with manually defined connections, **not** proof of autonomous weight selection, learned cross-model integration, or novel generalization. Task identities are defined by the two output coordinates. This experiment does not yet use the RES `.res` loader or validate its portability format.

To verify, run the script and check its JSON metrics and exact-agreement assertion.
