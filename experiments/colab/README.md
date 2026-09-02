# Google Colab Independent Validation Package

This directory contains the specification, expected benchmarks, and instructions for executing independent model validation in Google Colab or any clean external Python 3.10+ runtime.

## Files
- `reproducibility_spec.json`: Machine-readable specification with dataset checksum, split boundaries, 172-feature manifest, and tolerances.
- `expected_metrics.csv`: Ground-truth metrics from local Phase 11 evaluation.
- `notebooks/phase12_colab_validation.ipynb`: Standalone executable notebook.

## How to Execute on Google Colab
1. Open [Google Colab](https://colab.research.google.com).
2. Upload `notebooks/phase12_colab_validation.ipynb`.
3. When prompted in Cell 1, upload `data/features/freight_features_expanded.csv`.
4. Run all cells (`Runtime -> Run all`).
5. The notebook will:
   - Verify SHA-256 hash (`a998d58a6cd95d539b059f0877797f6f17a9dc94ac7ad8a2dbe30a79ae7b12ec`)
   - Check strict chronological boundaries
   - Fit Persistence, Ridge, XGBoost, and LSTM from scratch
   - Generate full reproducibility comparison tables against local metrics
   - Validate 5-fold walk-forward cross-validation
   - Test decision-layer signal quality against economic baseline strategies
   - Validate P10/P50/P90 uncertainty interval coverage
