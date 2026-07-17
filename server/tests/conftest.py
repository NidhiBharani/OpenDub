"""Test isolation: point the app at throwaway data/config dirs BEFORE any test module imports
`app` (and thus captures `app.config`'s module-level paths).

pytest imports conftest.py before collecting test modules, so this runs first regardless of test
filename order — without it, whichever test module imports `app` first (e.g. test_bench_metrics
via the bench package) would bind config to the real data dir and the real settings.yaml, breaking
the mock-provider e2e test.
"""
import os
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="opendub-tests-"))
os.environ.setdefault("OPENDUB_DATA_DIR", str(_TMP / "data"))
os.environ.setdefault("OPENDUB_CONFIGS_DIR", str(_TMP / "configs"))
