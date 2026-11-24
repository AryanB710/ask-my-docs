from pathlib import Path

import pytest

from askdocs.config import load_settings
from askdocs.ingest.index import build_index

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def settings():
    return load_settings(
        ROOT / "configs/offline.yaml",
        {
            "corpus_dir": str(ROOT / "data/demo/corpus"),
            "index_dir": "/nonexistent-unused",
            "observability.trace_dir": None,
            "observability.pricing_file": str(ROOT / "configs/pricing.yaml"),
        },
    )


@pytest.fixture(scope="session")
def index(settings):
    return build_index(settings)
