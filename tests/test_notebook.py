import ast
import json
from dataclasses import fields
from pathlib import Path

from segment.train import TrainConfig

NOTEBOOK = Path(__file__).parent.parent / "notebooks" / "train_kaggle.ipynb"


def _notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _source(cell: dict) -> str:
    return "".join(cell["source"])


def test_notebook_is_clean():
    nb = _notebook()
    assert nb["nbformat"] == 4
    assert nb["cells"]
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            assert cell["execution_count"] is None


def test_train_dict_matches_trainconfig_defaults():
    config_cell = next(
        _source(c)
        for c in _notebook()["cells"]
        if c["cell_type"] == "code" and "TRAIN =" in _source(c)
    )
    train = next(
        ast.literal_eval(node.value)
        for node in ast.parse(config_cell).body
        if isinstance(node, ast.Assign) and node.targets[0].id == "TRAIN"
    )
    defaults = {f.name: f.default for f in fields(TrainConfig)}
    assert set(train) <= set(defaults)
    for name, value in train.items():
        assert value == defaults[name], name
    assert set(train) == {
        "encoder",
        "encoder_weights",
        "epochs",
        "batch_size",
        "lr",
        "weight_decay",
        "crop",
        "patience",
        "seed",
    }


def test_repo_ref_is_a_placeholder_the_install_cell_rejects():
    sources = [_source(c) for c in _notebook()["cells"] if c["cell_type"] == "code"]
    assert any('REPO_REF = "<fill in' in s for s in sources)
    assert any('assert not REPO_REF.startswith("<")' in s for s in sources)
