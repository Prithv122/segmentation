import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))

from make_fixture import make_fixture


@pytest.fixture(scope="session")
def fixture_manifest(tmp_path_factory) -> Path:
    """Manifest of the synthetic chip set; chips live next to it under ``chips/``."""
    return make_fixture(tmp_path_factory.mktemp("fixture"))
