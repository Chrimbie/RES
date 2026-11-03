# RES — Reusable Executable Solutions

RES is an **experimental** project for extracting learned solutions from datasets, packaging them as portable `.res` files, and executing them through a lightweight head without loading the original training-model implementation.

## Working components

- **`res_solution_extractor.py`**: trains a fixed random ReLU feature basis and fits its output weights by ridge regression, starting from labeled numeric data.
- **`res_head.py`**: validates a self-describing `.res` archive, verifies its SHA-256 tensor payload, and executes predictions directly using NumPy.
- **`tests/test_res.py`**: checks transfer, prediction equivalence, input-shape validation, and digits accuracy.

## Quick start

```bash
python -m pip install -r requirements.txt
python res_solution_extractor.py --demo-digits --out digits.res
python res_head.py digits.res
python -m unittest discover -s tests -v
```

To classify a NumPy feature matrix:

```bash
python res_head.py digits.res --input samples.npy
```

For a numeric CSV with a header and class labels in the last column:

```bash
python res_solution_extractor.py --csv data.csv --out my_solution.res
```

## Format

The current v1.0 solution is a ZIP archive containing `manifest.json` and `weights.npz`. Its declared operator is `dense-relu-dense`; it includes input normalization, output class labels, and SHA-256 payload integrity.

## Scientific scope

The original training-model object is not required at inference time, **but the head still performs the mathematics represented by the solution**. This is not a proof of universal cross-architecture portability, architecture-free inference, or novel intelligence. Current extraction supports fixed-length numerical classification data only. Results depend on the dataset and split. Medical datasets are research benchmarks, not clinical validation.

## Roadmap

- Validated adapters for differing feature schemas and execution operations
- Additional solution types, including regression and retrieval
- Reproducible benchmarks against optimized inference runtimes
- Resource limits and security hardening for untrusted archives

Licensed under the repository's existing Apache 2.0 license.
