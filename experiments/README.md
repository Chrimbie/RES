# Historical experiments and compatibility

This directory preserves the reproducible compatibility-helper experiment and the cancer dataset test. The `results/` directory contains **historical measured reports**, not results re-run automatically by this commit.

The original experiments used different versions of the RES Head and extractor. Historical `.res` archives may not be readable by the repository's current v1.0 head without a format adapter.

- `res_compatibility_experiment.py`: feature ordering, documented inverse preprocessing, missing-feature rejection, and precision conversion.
- `test_res_cancer.py`: trains and transfers a new breast-cancer classification solution with the current repository API.
- `results/res_multilayer_compat_results.json`: historical multilayer PyTorch adapter benchmark (experimental script not yet included).
- `results/benchmark_original_vs_res_results.json`: historical scikit-learn vs NumPy head latency, not a comparison against optimized ONNX or PyTorch runtimes.

**Important:** Exact transfer means identical predictions for the same mathematical computation and input schema. It does not imply arbitrary cross-task or cross-architecture reuse. Cancer classification here is an educational benchmark, not a medical device.
