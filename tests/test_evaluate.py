import csv
import json

import pytest

from segment.evaluate import summarise, write_summary
from segment.metrics import Counts, bootstrap_ci, dice, iou
from segment.predict import PER_CHIP_COLUMNS

# Rows: (split, chip_id, scene, tp, fp, fn); tn is a constant filler.
# Four test scenes, two chips each.
#
# unet, test (scene: chips -> pooled tp/fp/fn -> IoU):
#   S1: t1 80/10/10 + t2 40/10/10 -> 120/20/20 -> 0.75
#   S2: t3 60/20/20 + t4 20/0/0   -> 80/20/20  -> 0.666667
#   S3: t5 50/10/10 + t6 50/0/0   -> 100/10/10 -> 0.833333
#   S4: t7 30/10/10 + t8 10/10/10 -> 40/20/20  -> 0.5
#   pooled tp=340 fp=70 fn=70 -> IoU 340/480 = 0.708333, Dice 680/820, precision = recall = 340/410
#   chip IoUs 0.8, 0.666667, 0.6, 1.0, 0.714286, 1.0, 0.6, 0.333333 -> macro IoU 0.714286
# ndwi_otsu, test:
#   S1: t1 60/20/30 + t2 30/20/20 -> 90/40/50
#   S2: t3 40/30/40 + t4 10/10/10 -> 50/40/50
#   S3: t5 30/20/30 + t6 30/10/10 -> 60/30/40
#   S4: t7 20/20/20 + t8 10/10/10 -> 30/30/30
#   pooled tp=230 fp=140 fn=170 -> IoU 230/540 = 0.425926
# gap_test = 0.708333 - 0.425926 = 0.282407
# unet, val: v1 90/10/10, v2 70/10/10 (V1), v3 40/0/0 (V2) -> pooled 200/20/20 -> IoU 0.833333
# ndwi_otsu, val: v1 50/30/30, v2 50/30/30 (V1), v3 50/0/0 (V2) -> 150/60/60 -> IoU 0.555556
# drop_unet = 0.833333 - 0.708333 = 0.125; drop_ndwi_otsu = 0.555556 - 0.425926 = 0.129630
UNET_TEST = [
    ("test", "t1", "S1", 80, 10, 10),
    ("test", "t2", "S1", 40, 10, 10),
    ("test", "t3", "S2", 60, 20, 20),
    ("test", "t4", "S2", 20, 0, 0),
    ("test", "t5", "S3", 50, 10, 10),
    ("test", "t6", "S3", 50, 0, 0),
    ("test", "t7", "S4", 30, 10, 10),
    ("test", "t8", "S4", 10, 10, 10),
]
NDWI_TEST = [
    ("test", "t1", "S1", 60, 20, 30),
    ("test", "t2", "S1", 30, 20, 20),
    ("test", "t3", "S2", 40, 30, 40),
    ("test", "t4", "S2", 10, 10, 10),
    ("test", "t5", "S3", 30, 20, 30),
    ("test", "t6", "S3", 30, 10, 10),
    ("test", "t7", "S4", 20, 20, 20),
    ("test", "t8", "S4", 10, 10, 10),
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
    assert g["n_chips"] == 8 and g["n_scenes"] == 4
    assert g["counts"] == {"tp": 340, "fp": 70, "fn": 70, "tn": 8000}
    assert g["iou"] == pytest.approx(340 / 480)
    assert g["dice"] == pytest.approx(680 / 820)
    assert g["precision"] == pytest.approx(340 / 410)
    assert g["recall"] == pytest.approx(340 / 410)
    chip_ious = [0.8, 2 / 3, 0.6, 1.0, 50 / 70, 1.0, 0.6, 1 / 3]
    assert g["macro_iou"] == pytest.approx(sum(chip_ious) / 8)
    assert g["n_nan_chips"] == 0
    assert g["per_scene"]["S1"] == {"n_chips": 2, "iou": pytest.approx(0.75)}
    assert g["per_scene"]["S2"] == {"n_chips": 2, "iou": pytest.approx(80 / 120)}
    assert g["per_scene"]["S3"]["iou"] == pytest.approx(100 / 120)
    assert g["per_scene"]["S4"]["iou"] == pytest.approx(0.5)
    assert s["groups"]["ndwi_otsu"]["test"]["iou"] == pytest.approx(230 / 540)
    val = s["groups"]["unet"]["val"]
    assert val["iou"] == pytest.approx(200 / 240) and val["n_scenes"] == 2


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
    assert g["n_chips"] == 3 and g["n_scenes"] == 1
    assert g["n_nan_chips"] == 1
    assert g["macro_iou"] == pytest.approx((0.5 + 1.0) / 2)
    assert g["iou"] == pytest.approx(150 / 200)


def test_paired_gap_point_and_pairing(tmp_path):
    extra = [*UNET_TEST, ("test", "t9", "S5", 5, 5, 5)]
    s = summarise(make_results(tmp_path, unet=UNET_VAL + extra))
    gap = s["comparisons"]["gap_test"]
    assert gap["n_chips"] == 8 and gap["n_scenes"] == 4
    assert gap["point"] == pytest.approx(340 / 480 - 230 / 540)


def test_drops(tmp_path):
    s = summarise(make_results(tmp_path))
    d = s["comparisons"]["drop_unet"]
    assert d["drop"] == pytest.approx(0.125)
    assert s["comparisons"]["drop_ndwi_otsu"]["drop"] == pytest.approx(150 / 270 - 230 / 540)


def test_comparisons_absent_when_groups_missing(tmp_path):
    s = summarise(make_results(tmp_path, unet=UNET_TEST, ndwi=NDWI_VAL))
    assert s["comparisons"] == {"gap_test": None, "drop_unet": None, "drop_ndwi_otsu": None}


def test_unknown_method_is_summarised(tmp_path):
    d = make_results(tmp_path)
    write_csv(d / "per_chip_other.csv", "mystery", UNET_TEST)
    assert summarise(d)["groups"]["mystery"]["test"]["n_chips"] == 8


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
    perfect = [("test", f"t{i}", f"S{i % 2}", 100, 0, 0) for i in range(4)]
    weak = [("test", f"t{i}", f"S{i % 2}", 50, 25, 25) for i in range(4)]
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
    perfect = [("test", f"t{i}", f"S{i}", 100, 0, 0) for i in range(3)]
    weak = [("test", f"t{i}", f"S{i}", 50, 25, 25) for i in range(3)]
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
    assert "| unet | test | 8 | 4 | 0.708 [" in text
    assert "Gap on test" in text and "0.282" in text
    assert "Val-to-test drop (unet): 0.125" in text
    assert "H1:" in text and "H2: within bound" in text
    assert "CI lower bound >= min_gap_iou" in text
    assert "scene-bootstrap" in text
    assert "## Per-scene IoU (test)" in text
    assert "| S1 | 0.750 (n=2) |" in text


def test_bootstrap_unit_is_recorded(tmp_path):
    s = summarise(make_results(tmp_path), n_boot=300)
    assert (s["bootstrap_unit"], s["seed"], s["n_boot"]) == ("scene", 0, 300)
    assert "scene-bootstrap" in s["rules"]["H1"] and "300" in s["rules"]["H1"]


def test_group_ci_resamples_scenes(tmp_path):
    s = summarise(make_results(tmp_path))
    scenes = [Counts(120, 20, 20, 2000), Counts(80, 20, 20, 2000)]
    scenes += [Counts(100, 10, 10, 2000), Counts(40, 20, 20, 2000)]
    lo, hi = bootstrap_ci(scenes, iou, n_boot=2000, seed=0)
    assert s["groups"]["unet"]["test"]["iou_ci"] == pytest.approx([lo, hi])


def test_scene_is_the_resampling_unit(tmp_path):
    # Three chips with very different IoU: in ONE scene they always move together, so every
    # replicate has the same pooled IoU and the CIs collapse. In three scenes they do not.
    chips = [(100, 0, 0), (10, 40, 40), (50, 25, 25)]
    base = [("test", f"c{i}", "S", *c) for i, c in enumerate(chips)]
    split = [("test", f"c{i}", f"S{i}", *c) for i, c in enumerate(chips)]
    weak = [("test", f"c{i}", "S", 1, 10, 10) for i in range(3)]
    weak_split = [("test", f"c{i}", f"S{i}", 1, 10, 10) for i in range(3)]

    one = summarise(make_results(tmp_path / "one", unet=base, ndwi=weak))
    g = one["groups"]["unet"]["test"]
    assert g["n_chips"] == 3 and g["n_scenes"] == 1
    assert g["iou_ci"][0] == g["iou_ci"][1] == pytest.approx(g["iou"])
    gap = one["comparisons"]["gap_test"]
    assert gap["ci"][0] == gap["ci"][1] == pytest.approx(gap["point"])

    many = summarise(make_results(tmp_path / "many", unet=split, ndwi=weak_split))
    lo, hi = many["groups"]["unet"]["test"]["iou_ci"]
    assert lo < hi
    glo, ghi = many["comparisons"]["gap_test"]["ci"]
    assert glo < ghi
