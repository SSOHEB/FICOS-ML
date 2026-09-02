"""Tests for Phase 12: Independent Colab Validation Preparation.

Verifies:
1. All Colab package artifacts exist:
   - notebooks/phase12_colab_validation.ipynb
   - experiments/colab/expected_metrics.csv
   - experiments/colab/reproducibility_spec.json
   - experiments/colab/README.md
   - reports/phase12_colab_validation_plan.md
2. Reproducibility spec schema: valid SHA-256, correct dataset shape (3353, 299), 172 features, 4 targets.
3. Jupyter notebook syntax and structure: valid JSON, contains code cells for data loading, checksum, training, comparison table, decision layer, and uncertainty calibration.
4. Expected metrics consistency with Phase 11 model audit.
"""

from pathlib import Path
import json
import hashlib
import pandas as pd
import pytest


def test_colab_artifacts_existence():
    """Verify all expected Phase 12 artifacts exist and are non-empty."""
    root = Path(".")
    expected_files = [
        root / "notebooks" / "phase12_colab_validation.ipynb",
        root / "experiments" / "colab" / "expected_metrics.csv",
        root / "experiments" / "colab" / "reproducibility_spec.json",
        root / "experiments" / "colab" / "README.md",
        root / "reports" / "phase12_colab_validation_plan.md",
    ]
    for p in expected_files:
        assert p.exists(), f"Missing required Phase 12 file: {p}"
        assert p.stat().st_size > 100, f"File {p} is empty or unexpectedly small"


def test_reproducibility_spec_schema_and_hashes():
    """Verify metadata in reproducibility_spec.json matches current expanded feature matrix."""
    spec_path = Path("experiments/colab/reproducibility_spec.json")
    with open(spec_path, "r", encoding="utf-8") as f:
        spec = json.load(f)

    feature_file = Path("data/features/freight_features_expanded.csv")
    with open(feature_file, "rb") as f:
        actual_hash = hashlib.sha256(f.read()).hexdigest()

    assert spec["dataset_sha256"] == actual_hash, "Spec SHA-256 does not match physical feature dataset"
    assert spec["dataset_shape"] == [3353, 299], f"Unexpected shape in spec: {spec['dataset_shape']}"
    assert spec["feature_count"] == 172, f"Expected 172 features, found {spec['feature_count']}"
    assert len(spec["features"]) == 172
    assert len(spec["targets"]) == 4

    # Date ranges
    assert spec["train_rows"] == 1107
    assert spec["val_rows"] == 237
    assert spec["test_rows"] == 238
    assert spec["train_date_range"] == ["2020-02-06", "2024-08-28"]
    assert spec["val_date_range"] == ["2024-08-29", "2025-08-28"]
    assert spec["test_date_range"] == ["2025-08-29", "2026-08-31"]


def test_colab_notebook_structure():
    """Verify notebooks/phase12_colab_validation.ipynb is valid JSON and contains required cells."""
    nb_path = Path("notebooks/phase12_colab_validation.ipynb")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    assert "cells" in nb
    assert len(nb["cells"]) >= 15, "Notebook has insufficient cells"

    all_code = "".join([
        "".join(cell.get("source", []))
        for cell in nb["cells"]
        if cell.get("cell_type") == "code"
    ])

    # Check key functionalities are implemented in notebook code
    assert "EXPECTED_HASH" in all_code, "Notebook missing SHA-256 assertion"
    assert "RidgePipeline" in all_code or "Ridge" in all_code, "Notebook missing Ridge pipeline"
    assert "XGBoost" in all_code or "xgb" in all_code, "Notebook missing XGBoost"
    assert "LSTM" in all_code, "Notebook missing LSTM"
    assert "REPRODUCIBILITY" in all_code, "Notebook missing reproducibility comparison"
    assert "P10" in all_code and "P90" in all_code, "Notebook missing uncertainty calibration"
    assert "CHARTER NOW" in all_code or "charter_now" in all_code, "Notebook missing decision logic"


def test_expected_metrics_table_consistency():
    """Verify expected_metrics.csv matches Phase 11 reported values."""
    p = Path("experiments/colab/expected_metrics.csv")
    df = pd.read_csv(p)

    assert len(df) == 16, f"Expected 16 model-vessel rows, found {len(df)}"
    assert set(df["model"].unique()) == {"Persistence", "Ridge", "XGBoost", "LSTM"}
    assert set(df["vessel"].unique()) == {"Handy", "Supramax", "Panamax", "Cape"}

    # Spot-check known benchmarks
    cape_ridge = df[(df["vessel"] == "Cape") & (df["model"] == "Ridge")].iloc[0]
    assert cape_ridge["mae"] == 1098.31
    assert cape_ridge["da_pct"] == 62.61

    panamax_ridge = df[(df["vessel"] == "Panamax") & (df["model"] == "Ridge")].iloc[0]
    assert panamax_ridge["mae"] == 238.48
    assert panamax_ridge["da_pct"] == 63.87
