"""Classical-vs-CNN comparison: statistics checked by hand + a full run on synthetic fixture data."""
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from cnn.segmentation.compare import bootstrap_ci, paired, score, summarize


def test_paired_wins_ties_losses_by_hand():
    cnn = [0.9, 0.8, 0.7, 0.5, 0.5]
    cls = [0.5, 0.8, 0.705, 0.9, 0.3]
    p = paired(cnn, cls, n_boot=2000)
    assert (p["cnn_better"], p["ties"], p["classical_better"]) == (2, 2, 1)
    assert p["mean_difference"] == pytest.approx(np.mean(np.array(cnn) - np.array(cls)), abs=1e-4)
    assert p["ci95"][0] <= p["mean_difference"] <= p["ci95"][1] and p["n"] == 5


def test_paired_identical_methods_has_no_test_statistic():
    p = paired([0.5, 0.6, 0.7], [0.5, 0.6, 0.7])
    assert p["mean_difference"] == 0 and p["wilcoxon_p"] is None and p["ties"] == 3


def test_clear_difference_gives_small_p_and_ci_above_zero():
    rng = np.random.default_rng(1)
    cls = rng.uniform(0.3, 0.6, 60); cnn = cls + rng.uniform(0.2, 0.3, 60)
    p = paired(cnn, cls, n_boot=2000)
    assert p["wilcoxon_p"] < 1e-6 and p["ci95"][0] > 0.15 and p["cnn_better"] == 60


def test_bootstrap_ci_contains_mean_and_is_deterministic():
    v = np.random.default_rng(0).normal(0.7, 0.1, 100)
    lo, hi = bootstrap_ci(v, 2000, seed=3)
    assert lo < v.mean() < hi and (lo, hi) == bootstrap_ci(v, 2000, seed=3) and hi - lo < 0.1
    s = summarize([0.5, None, 0.7], 500)
    assert s["n"] == 2 and s["mean"] == 0.6


def test_score_missing_mask_counts_as_empty_and_resizes_nearest():
    ref = np.zeros((40, 60), bool); ref[10:30, 20:50] = True
    none = score(None, ref)
    assert none["dice"] == 0.0 and none["found"] is False and none["pixel_accuracy"] == pytest.approx(1 - ref.mean())
    small = np.zeros((20, 30), np.uint8); small[5:15, 10:25] = 255                  # same lesion at half size
    s = score(small, ref)
    assert s["found"] and s["dice"] == pytest.approx(1.0, abs=1e-6)


# ------------------------------------------------------------------ full run
tf = pytest.importorskip("tensorflow")
from cnn.segmentation import compare, train as seg_train
from tests.cnn_fixtures import write_dataset


@pytest.fixture(scope="module")
def fixture_run():
    root = Path(tempfile.mkdtemp(prefix="melanolens_cmp_"))
    write_dataset(root / "isic", n=60, size=96)
    models = root / "models"; models.mkdir()
    img, msk = str(root / "isic" / "images"), str(root / "isic" / "masks")
    seg_train.main(["--images-dir", img, "--masks-dir", msk, "--model-path", str(models / "u.keras"), "--img-size", "64",
                    "--base-filters", "8", "--depth", "3", "--epochs", "4", "--batch-size", "8", "--no-eval"])
    return {"models": models, "images": img, "masks": msk}


def test_full_comparison_protocol(fixture_run):
    f = fixture_run
    sp = json.loads((f["models"] / "u_split.json").read_text())
    r = compare.compare(f["models"] / "u.keras", f["images"], f["masks"], n_boot=300)
    assert r["evaluated_on"] == "held-out test split" and r["n_test"] == len(sp["test"])
    assert [x["id"] for x in r["per_image"]] == sp["test"]                                    # exactly the test images, in order
    sel = r["classical"]["selection"]
    assert sel["selected_on"] == "validation split" and sel["n_validation"] == len(sp["val"])
    assert r["classical"]["headline_method"] == max(sel["validation_mean_dice"], key=sel["validation_mean_dice"].get)
    assert len(r["classical_all_methods"]) == 10 and r["paired"]["dice"]["n"] == r["n_test"]
    n = r["paired"]["dice"]
    assert n["cnn_better"] + n["ties"] + n["classical_better"] == r["n_test"]
    assert (f["models"] / "u_comparison.json").exists()
    # the headline mean is recomputable from the per-image rows
    assert r["headline"]["cnn"]["dice"]["mean"] == pytest.approx(np.mean([x["cnn_dice"] for x in r["per_image"]]), abs=2e-3)


def test_combo_override_is_labelled_as_not_data_selected(fixture_run):
    f = fixture_run
    r = compare.compare(f["models"] / "u.keras", f["images"], f["masks"], combo="sobel:conventional", max_images=4, n_boot=200,
                        out_path=f["models"] / "x.json")
    assert r["classical"]["headline_method"] == "sobel:conventional" and "not selected from data" in r["classical"]["selection"]["selected_on"]
    with pytest.raises(Exception, match="--combo"):
        compare.compare(f["models"] / "u.keras", f["images"], f["masks"], combo="nope", out_path=f["models"] / "y.json")
