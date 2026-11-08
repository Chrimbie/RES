# Frozen-weight composition: reproducible evidence

Run `python experiments/frozen_weight_composition.py` after installing `requirements.txt`.

Two independently trained 2-input, 8-hidden-unit MLP classifiers learn separate synthetic binary tasks: sign of x-coordinate and sign of y-coordinate. Each uses 3,000 training samples and 1,000 held-out samples from the same seeded input matrix. A third network is constructed by copying both models' frozen parameters into one 16-hidden-unit, two-output network, with block-sparse output connections. No third-model training or runtime task selector is used; both branches compute on every input.

Measured in the test environment:
| Metric | Accuracy |
| --- | ---: |
| Original A | 99.7% |
| Original B | 99.5% |
| Composed output A | 99.7% |
| Composed output B | 99.5% |
| Both outputs correct on same example | 99.2% |

The composed network's two predictions agreed exactly with the original two networks on all 1,000 held-out inputs.

**Interpretation:** this is a controlled demonstration of *weight-preserving network composition* for compatible architectures. It is **not** evidence of autonomous weight discovery, learned cross-model interactions, new knowledge, or general compatibility with arbitrary pretrained models. The original task rules are distinct but both outputs are always computed. No RES `.res` loader integration is claimed. This experiment does not modify RES's existing implementation.
