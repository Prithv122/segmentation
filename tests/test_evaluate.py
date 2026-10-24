import csv
import json

import pytest

from segment.evaluate import summarise, write_summary
from segment.metrics import Counts, dice, iou
from segment.predict import PER_CHIP_COLUMNS

# Rows: (split, chip_id, scene, tp, fp, fn); tn is a constant filler.
#
# unet, test:  t1 80/10/10, t2 40/10/10 (scene S1); t3 60/20/20, t4 20/0/0 (scene S2)
#   pooled tp=200 fp=40 fn=40 -> IoU 200/280 = 0.714286, Dice 400/480, precision = recall = 200/240
#   scene S1 120/20/20 -> 0.75; scene S2 80/20/20 -> 80/120 = 0.666667
#   macro IoU = mean(0.8, 0.666667, 0.6, 1.0) = 0.766667
# ndwi_otsu, test: t1 60/20/30, t2 30/20/20, t3 40/30/40, t4 10/10/10
#   pooled tp=140 fp=80 fn=100 -> IoU 140/320 = 0.4375
# gap_test = 0.714286 - 0.4375 = 0.276786
# unet, val: v1 90/10/10, v2 70/10/10, v3 40/0/0 -> pooled 200/20/20 -> IoU 200/240 = 0.833333
# ndwi_otsu, val: v1 50/30/30, v2 50/30/30, v3 50/0/0 -> pooled 150/60/60 -> IoU 150/270 = 0.555556
# drop_unet = 0.833333 - 0.714286 = 0.119048; drop_ndwi_otsu = 0.555556 - 0.4375 = 0.118056
UNET_TEST = [
    ("test", "t1", "S1", 80, 10, 10),
    ("test", "t2", "S1", 40, 10, 10),
    ("test", "t3", "S2", 60, 20, 20),
    ("test", "t4", "S2", 20, 0, 0),
]
NDWI_TEST = [
    ("test", "t1", "S1", 60, 20, 30),
    ("test", "t2", "S1", 30, 20, 20),
    ("test", "t3", "S2", 40, 30, 40),
    ("test", "t4", "S2", 10, 10, 10),
]
UNET_VAL = [
    ("val", "v1", "V1", 90, 10, 10),
    ("val", "v2", "V1", 70, 10, 10),
    ("val", "v3", "V2", 40, 0, 0),
]
NDWI_VAL = [
    ("val", "v1", "V1", 50, 30, 30),
    ("val", "v2", "V1", 50, 30, 30),
    ("val", "v3", "V2", 50, 0, 0),
]


def write_csv(path, method, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(PER_CHIP_COLUMNS)
        for split, chip_id, scene, tp, fp, fn in rows:
            c = Counts(tp, fp, fn, 1000)
            w.writerow([method, split, chip_id, scene, tp, fp, fn, 1000, iou(c), dice(c)])


def make_results(tmp_path, unet=None, ndwi=None, prereg=None):
    d = tmp_path / "results"
    d.mkdir(parents=True)
    write_csv(d / "per_chip_unet.csv", "unet", UNET_VAL + UNET_TEST if unet is None else unet)
    write_csv(
        d / "per_chip_ndwi_otsu.csv", "ndwi_otsu", NDWI_VAL + NDWI_TEST if ndwi is None else ndwi
    )
    if prereg is not None:
        (d / "prereg.json").write_text(json.dumps(prereg), encoding="utf-8")
    return d


def test_pooled_metrics_and_scene_iou(tmp_path):
    s = summarise(make_results(tmp_path))
    g = s["groups"]["unet"]["test"]
    assert g["n_chips"] == 4
    assert g["counts"] == {"tp": 200, "fp": 40, "fn": 40, "tn": 4000}
    assert g["iou"] == pytest.approx(200 / 280)
    assert g["dice"] == pytest.approx(400 / 480)
    assert g["precision"] == pytest.approx(200 / 240)
    assert g["recall"] == pytest.approx(200 / 240)
    assert g["macro_iou"] == pytest.approx((0.8 + 2 / 3 + 0.6 + 1.0) / 4)
    assert g["n_nan_chips"] == 0
    assert g["per_scene"]["S1"] == {"n_chips": 2, "iou": pytest.approx(0.75)}
    assert g["per_scene"]["S2"] == {"n_chips": 2, "iou": pytest.approx(80 / 120)}
    assert s["groups"]["ndwi_otsu"]["test"]["iou"] == pytest.approx(0.4375)
    assert s["groups"]["unet"]["val"]["iou"] == pytest.approx(200 / 240)


def test_ci_contains_point_estimate(tmp_path):
    s = summarise(make_results(tmp_path))
    g = s["groups"]["unet"]["test"]
    lo, hi = g["iou_ci"]
    assert lo < hi
    assert lo <= g["iou"] <= hi
    glo, ghi = s["comparisons"]["gap_test"]["ci"]
    assert glo < ghi
    assert glo <= s["comparisons"]["gap_test"]["point"] <= ghi


def test_macro_excludes_nan_chips(tmp_path):
    rows = [
        ("test", "a", "S", 50, 50, 0),
        ("test", "b", "S", 0, 0, 0),
        ("test", "c", "S", 100, 0, 0),
    ]
    s = summarise(make_results(tmp_path, unet=rows, ndwi=rows))
    g = s["groups"]["unet"]["test"]
    assert g["n_chips"] == 3
    assert g["n_nan_chips"] == 1
    assert g["macro_iou"] == pytest.approx((0.5 + 1.0) / 2)
    assert g["iou"] == pytest.approx(150 / 200)


def test_paired_gap_point_and_pairing(tmp_path):
    extra = [*UNET_TEST, ("test", "t9", "S3", 5, 5, 5)]
    s = summarise(make_results(tmp_path, unet=UNET_VAL + extra))
    gap = s["comparisons"]["gap_test"]
    assert gap["n_chips"] == 4
    assert gap["point"] == pytest.approx(200 / 280 - 0.4375)


def test_drops(tmp_path):
    s = summarise(make_results(tmp_path))
    d = s["comparisons"]["drop_unet"]
    assert d["drop"] == pytest.approx(200 / 240 - 200 / 280)
    assert s["comparisons"]["drop_ndwi_otsu"]["drop"] == pytest.approx(150 / 270 - 0.4375)


def test_comparisons_absent_when_groups_missing(tmp_path):
    s = summarise(make_results(tmp_path, unet=UNET_TEST, ndwi=NDWI_VAL))
    assert s["comparisons"] == {"gap_test": None, "drop_unet": None, "drop_ndwi_otsu": None}


def test_unknown_method_is_summarised(tmp_path):
    d = make_results(tmp_path)
    write_csv(d / "per_chip_other.csv", "mystery", UNET_TEST)
    assert summarise(d)["groups"]["mystery"]["test"]["n_chips"] == 4


def test_no_csv_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        summarise(tmp_path)


def test_missing_prereg_gives_null_verdicts(tmp_path):
    s = summarise(make_results(tmp_path))
    assert s["prereg"] is None
    assert s["verdicts"] == {"H1": None, "H2": None}
    assert "H1" in s["rules"] and "H2" in s["rules"]


def test_h1_supported(tmp_path):
    # unet IoU 1.0 on every chip, baseline 50/25/25 -> 0.5 on every chip: every resample gives 0.5
    perfect = [("test", f"t{i}", "S", 100, 0, 0) for i in range(4)]
    weak = [("test", f"t{i}", "S", 50, 25, 25) for i in range(4)]
    d = make_results(
        tmp_path, unet=perfect, ndwi=weak, prereg={"min_gap_iou": 0.05, "max_drop_iou": 0.15}
    )
    s = summarise(d)
    assert s["comparisons"]["gap_test"]["point"] == pytest.approx(0.5)
    assert s["comparisons"]["gap_test"]["ci"] == pytest.approx([0.5, 0.5])
    assert s["verdicts"]["H1"] == "supported"


def test_h1_not_supported(tmp_path):
    d = make_results(
        tmp_path,
        unet=UNET_TEST,
        ndwi=UNET_TEST,
        prereg={"min_gap_iou": 0.05, "max_drop_iou": 0.15},
    )
    s = summarise(d)
    assert s["comparisons"]["gap_test"]["point"] == 0
    assert s["comparisons"]["gap_test"]["ci"] == [0.0, 0.0]
    assert s["verdicts"]["H1"] == "not supported"


def test_h1_inconclusive(tmp_path):
    base = summarise(make_results(tmp_path / "a"))
    lo, hi = base["comparisons"]["gap_test"]["ci"]
    assert lo < hi
    mid = (lo + hi) / 2
    d = make_results(tmp_path / "b", prereg={"min_gap_iou": mid, "max_drop_iou": 0.15})
    assert summarise(d)["verdicts"]["H1"] == "inconclusive"


def test_h1_boundary_lower_bound_equal_is_supported(tmp_path):
    perfect = [("test", f"t{i}", "S", 100, 0, 0) for i in range(3)]
    weak = [("test", f"t{i}", "S", 50, 25, 25) for i in range(3)]
    d = make_results(
        tmp_path, unet=perfect, ndwi=weak, prereg={"min_gap_iou": 0.5, "max_drop_iou": 1}
    )
    assert summarise(d)["verdicts"]["H1"] == "supported"


def test_h2_both_ways(tmp_path):
    ok = make_results(tmp_path / "a", prereg={"min_gap_iou": 0.05, "max_drop_iou": 0.15})
    assert summarise(ok)["verdicts"]["H2"] == "within bound"
    bad = make_results(tmp_path / "b", prereg={"min_gap_iou": 0.05, "max_drop_iou": 0.10})
    assert summarise(bad)["verdicts"]["H2"] == "poor generalisation"


def test_prereg_missing_key_raises(tmp_path):
    d = make_results(tmp_path, prereg={"min_gap_iou": 0.05})
    with pytest.raises(ValueError):
        summarise(d)


def test_nan_becomes_none_in_json(tmp_path):
    rows = [("test", "z", "S", 0, 0, 0)]
    d = make_results(tmp_path, unet=rows, ndwi=rows)
    s = summarise(d)
    g = s["groups"]["unet"]["test"]
    assert g["iou"] is None and g["macro_iou"] is None and g["iou_ci"] == [None, None]
    json_path, md_path = write_summary(s, d)

    def no_constants(name):
        raise AssertionError(f"non-finite constant {name} in JSON")

    loaded = json.loads(json_path.read_text(encoding="utf-8"), parse_constant=no_constants)
    assert loaded["groups"]["unet"]["test"]["iou"] is None
    assert "n/a" in md_path.read_text(encoding="utf-8")


def test_write_summary_deterministic_and_lf(tmp_path):
    prereg = {"min_gap_iou": 0.05, "max_drop_iou": 0.15}
    d = make_results(tmp_path, prereg=prereg)
    j1, m1 = write_summary(summarise(d), d)
    first = (j1.read_bytes(), m1.read_bytes())
    j2, m2 = write_summary(summarise(d), d)
    assert (j2.read_bytes(), m2.read_bytes()) == first
    assert b"\r" not in first[0] and b"\r" not in first[1]
    loaded = json.loads(first[0])
    assert list(loaded) == sorted(loaded)


def test_markdown_content(tmp_path):
    d = make_results(tmp_path, prereg={"min_gap_iou": 0.05, "max_drop_iou": 0.15})
    _, md = write_summary(summarise(d), d)
    text = md.read_text(encoding="utf-8")
    assert "| unet | test | 4 | 0.714 [" in text
    assert "Gap on test" in text and "0.277" in text
    assert "Val-to-test drop (unet): 0.119" in text
    assert "H1:" in text and "H2: within bound" in text
    assert "CI lower bound >= min_gap_iou" in text
    assert "## Per-scene IoU (test)" in text
    assert "| S1 | 0.750 (n=2) |" in text
