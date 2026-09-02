"""Test package imports and third-party dependencies."""

import importlib
import pytest


def test_internal_package_imports():
    """Verify that all core src submodules are properly importable."""
    modules = [
        "src",
        "src.data",
        "src.features",
        "src.models",
        "src.evaluation",
        "src.utils",
    ]
    for mod in modules:
        imported = importlib.import_module(mod)
        assert imported is not None, f"Failed to import {mod}"


def test_core_ml_dependencies_importable():
    """Verify that essential ML and time-series libraries are installed and importable."""
    dependencies = [
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "statsmodels",
        "yaml",
        "matplotlib",
        "seaborn",
    ]
    for dep in dependencies:
        imported = importlib.import_module(dep)
        assert imported is not None, f"Failed to import third-party dependency: {dep}"
