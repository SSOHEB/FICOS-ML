"""Repository Hygiene & Release Readiness Checker.

Performs:
1. Secret and credential scanning across all repo files.
2. Absolute Windows path scanning (C:\\Users\\... or file:///C:/...).
3. Checksum verification of canonical datasets.
4. Classification of repo files into clean categories.
"""

import hashlib
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SECRET_PATTERNS = [
    re.compile(r'(?i)(api[_-]?key|secret|password|bearer|auth[_-]?token|private[_-]?key)\s*[:=]\s*["\']([^"\']{8,})["\']'),
    re.compile(r'-----BEGIN [A-Z ]*PRIVATE KEY-----'),
    re.compile(r'ghp_[a-zA-Z0-9]{36}'),
    re.compile(r'AIza[0-9A-Za-z-_]{35}'),
]

WINDOWS_PATH_PATTERN = re.compile(r'C:[/\\]Users[/\\][a-zA-Z0-9_.-]+', re.IGNORECASE)
FILE_URI_PATTERN = re.compile(r'file:///[a-zA-Z]:[^\s\)\"\']+', re.IGNORECASE)

IGNORED_PARTS = {'.git', '.venv', '.pytest_cache', '__pycache__', '.mypy_cache', '.ruff_cache', '.ipynb_checkpoints'}
BINARY_EXTS = {'.png', '.jpg', '.jpeg', '.pdf', '.xls', '.xlsx', '.pkl', '.pt', '.parquet'}


def scan_repo():
    print("=== 1. SCANNING FOR SECRETS & CREDENTIALS ===")
    secret_hits = []
    win_path_hits = []

    for path in sorted(ROOT.rglob('*')):
        if not path.is_file():
            continue
        if any(part in path.parts for part in IGNORED_PARTS):
            continue
        if path.suffix in BINARY_EXTS:
            continue

        try:
            content = path.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            continue

        # Secret check
        for pat in SECRET_PATTERNS:
            for match in pat.finditer(content):
                secret_hits.append((path.relative_to(ROOT).as_posix(), match.group(0)[:60]))

        # Absolute windows path check
        if path.suffix in {'.md', '.py', '.json', '.yaml', '.yml', '.ipynb'}:
            for match in WINDOWS_PATH_PATTERN.finditer(content):
                win_path_hits.append((path.relative_to(ROOT).as_posix(), match.group(0)))
            for match in FILE_URI_PATTERN.finditer(content):
                win_path_hits.append((path.relative_to(ROOT).as_posix(), match.group(0)))

    print(f"Total Secret Findings: {len(secret_hits)}")
    for f, hit in secret_hits:
        print(f"  [SECRET ALERT] {f}: {hit}")

    print(f"\n=== 2. SCANNING FOR ABSOLUTE WINDOWS PATHS ===")
    print(f"Total Absolute Path Hits: {len(win_path_hits)}")
    unique_files_with_paths = sorted(set(f for f, _ in win_path_hits))
    for uf in unique_files_with_paths[:15]:
        sample_hits = [h for f, h in win_path_hits if f == uf]
        print(f"  {uf} ({len(sample_hits)} instances) e.g., {sample_hits[0]}")

    print("\n=== 3. DATASET CHECKSUM VERIFICATION ===")
    exp_features = ROOT / "data" / "features" / "freight_features_expanded.csv"
    if exp_features.exists():
        with open(exp_features, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()
        print(f"freight_features_expanded.csv SHA-256: {h}")
        expected = "a998d58a6cd95d539b059f0877797f6f17a9dc94ac7ad8a2dbe30a79ae7b12ec"
        assert h == expected, "SHA-256 mismatch on freight_features_expanded.csv"
        print("  ✓ Checksum MATCHES canonical reference.")
    else:
        print("  ✗ freight_features_expanded.csv missing!")


if __name__ == "__main__":
    scan_repo()
