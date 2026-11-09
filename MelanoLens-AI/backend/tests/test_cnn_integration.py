"""Train tiny models on SYNTHETIC fixture data, then test save/load, inference, API, evaluation files.

These verify the machinery. They say nothing about accuracy on real ISIC images.
"""
import io
import json
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

pytest.importorskip("tensorflow")

from app import create_app
from cnn import registry, utils
from cnn.classification import evaluate as cls_eval, predict as cls_predict, train as cls_train
from cnn.errors import ModelNotAvailable
from cnn.segmentation import evaluate as seg_eval, predict as seg_predict, train as seg_train
from config import Config
from services import synthetic
from tests.cnn_fixtures import make_case, write_dataset
from tests.helpers import png_bytes, upload


@pytest.fixture(scope="module")
def trained():
    root = Path(tempfile.mkdtemp(prefix="melanolens_cnn_"))
    write_dataset(root / "isic", n=96, size=96)
    models = root / "models"; models.mkdir()
    img, msk = str(root / "isic" / "images"), str(root / "isic" / "masks")
    seg_train.main(["--images-dir", img, "--masks-dir", msk, "--model-path", str(models / "u.keras"), "--img-size", "64",
                    "--base-filters", "8", "--depth", "3", "--epochs", "25", "--batch-size", "8", "--patience", "25"])
    cls_train.main(["--images-dir", img, "--model-path", str(models / "c.keras"), "--img-size", "96", "--alpha", "0.35",
                    "--weights", "none", "--epochs", "2", "--fine-tune-epochs", "0", "--batch-size", "16"])
    yield {"root": root, "models": models, "images": img, "masks": msk}
    shutil.rmtree(root, ignore_errors=True)


def _client(models, seg=True, cls=True, results=None):
    results = results or Path(tempfile.mkdtemp(prefix="melanolens_res_"))

    class Cfg(Config):
        TESTING = True
        RESULTS_DIR = results
        SEG_MODEL_PATH = models / ("u.keras" if seg else "missing_u.keras")
        CLS_MODEL_PATH = models / ("c.keras" if cls else "missing_c.keras")
    return create_app(Cfg).test_client(), results


def test_training_wrote_all_artifacts(trained):
    names = {p.name for p in trained["models"].iterdir()}
    for f in ("u.keras", "u_meta.json", "u_history.json", "u_split.json", "u_eval.json",
              "c.keras", "c_meta.json", "c_history.json", "c_split.json", "c_eval.json", "checkpoints"):
        assert f in names, f
    assert {p.name for p in (trained["models"] / "checkpoints").iterdir()} == {"u_best.keras", "c_best.keras"}


def test_test_split_is_held_out_and_disjoint(trained):
    for stem in ("u", "c"):
        sp = json.loads((trained["models"] / f"{stem}_split.json").read_text())
        assert not set(sp["train"]) & set(sp["test"]) and not set(sp["val"]) & set(sp["test"]) and not set(sp["train"]) & set(sp["val"])
        assert len(sp["test"]) > 0


def test_segmentation_model_reloads_and_matches_saved_metrics(trained):
    pred = seg_predict.SegmentationPredictor.load(trained["models"] / "u.keras")
    ev = json.loads((trained["models"] / "u_eval.json").read_text())
    assert ev["evaluated_on"] == "held-out test split" and ev["n_test"] > 0
    again = seg_eval.evaluate_model(trained["models"] / "u.keras", trained["images"], trained["masks"],
                                    out_path=trained["models"] / "u_eval2.json")
    assert again["metrics"]["dice"]["mean"] == ev["metrics"]["dice"]["mean"]       # reload -> identical numbers
    assert ev["metrics"]["dice"]["mean"] > 0.5                                       # it genuinely learned the synthetic task
    img, mask = make_case(5, 96, False)
    out = pred.predict(img)
    assert out["mask"].shape == img.shape[:2] and out["probability"].min() >= 0 and out["probability"].max() <= 1


def test_classifier_reload_probabilities_and_metadata(trained):
    pred = cls_predict.ClassificationPredictor.load(trained["models"] / "c.keras")
    assert pred.class_names == ["benign", "malignant"]
    img, _ = make_case(1, 96, False)
    r = pred.predict(img)
    assert abs(sum(r["probabilities"].values()) - 1) < 1e-3 and r["predicted_class"] in pred.class_names
    assert r["confidence"] == max(r["probabilities"].values()) and "NOT a medical diagnosis" in r["disclaimer"]
    ev = json.loads((trained["models"] / "c_eval.json").read_text())
    assert ev["evaluated_on"] == "held-out test split" and sum(map(sum, ev["confusion_matrix"]["matrix"])) == ev["n_test"]


def test_classifier_evaluation_uses_only_test_ids(trained):
    sp = json.loads((trained["models"] / "c_split.json").read_text())
    ev = json.loads((trained["models"] / "c_eval.json").read_text())
    assert ev["n_test"] == len(sp["test"]) and sum(ev["test_class_counts"].values()) == len(sp["test"])


def test_status_reports_missing_models_without_paths(trained):
    c, _ = _client(trained["models"], seg=False, cls=False)
    s = c.get("/api/cnn/status").get_json()
    assert s["segmentation"]["available"] is False and "not trained" in s["segmentation"]["reason"].lower()
    assert "/" not in json.dumps(s["segmentation"]["model_file"]) and str(trained["root"]) not in json.dumps(s)


def test_missing_model_returns_503_not_fake_output(trained):
    c, _ = _client(trained["models"], seg=False, cls=False)
    img, _m = make_case(2, 96, False)
    for ep in ("segment", "classify"):
        r = c.post(f"/api/cnn/{ep}", data={"image": (io.BytesIO(png_bytes(img)), "x.png")}, content_type="multipart/form-data")
        assert r.status_code == 503 and r.get_json()["error"]["code"] == "model_missing"
        assert str(trained["root"]) not in r.get_data(as_text=True)


def test_corrupt_model_file_is_reported(trained, tmp_path):
    (tmp_path / "u.keras").write_bytes(b"not a model")
    c, _ = _client(tmp_path, seg=True, cls=False)
    img, _m = make_case(2, 96, False)
    r = c.post("/api/cnn/segment", data={"image": (io.BytesIO(png_bytes(img)), "x.png")}, content_type="multipart/form-data")
    assert r.status_code == 503 and r.get_json()["error"]["code"] == "model_invalid"


def test_input_validation(trained):
    c, _ = _client(trained["models"])
    assert c.post("/api/cnn/segment", json={}).get_json()["error"]["code"] == "missing_file"
    assert c.post("/api/cnn/classify", json={"result_id": "0" * 32}).status_code == 404
    bad = c.post("/api/cnn/segment", data={"image": (io.BytesIO(b"hello"), "x.txt")}, content_type="multipart/form-data")
    assert bad.status_code == 400 and bad.get_json()["error"]["code"] == "invalid_file_type"


def test_segment_upload_mode_returns_inline_images_and_stores_nothing(trained):
    c, res = _client(trained["models"])
    img, _m = make_case(3, 96, True)
    r = c.post("/api/cnn/segment", data={"image": (io.BytesIO(png_bytes(img)), "x.png")}, content_type="multipart/form-data")
    j = r.get_json()
    assert r.status_code == 200 and j["result_id"] is None and set(j["images_inline"]) >= {"cnn_mask", "cnn_region", "cnn_overlay", "cnn_probability"}
    assert j["images_inline"]["cnn_mask"].startswith("data:image/png;base64,") and not any(res.iterdir())
    assert j["model"]["name"] == "U-Net" and "comparison" in j and j["comparison"]["classical_available"] is False


def test_full_workflow_classical_plus_cnn_with_reference(trained):
    c, res = _client(trained["models"])
    img, ref = make_case(7, 96, False)
    img = cv2.resize(img, (256, 256)); ref = cv2.resize(ref, (256, 256), interpolation=cv2.INTER_NEAREST)
    rid = upload(c, img, ref).get_json()["result_id"]
    assert c.post("/api/process", json={"result_id": rid, "params": {"selection": "closest_to_center"}}).status_code == 200
    seg = c.post("/api/cnn/segment", json={"result_id": rid}).get_json()
    cl = c.post("/api/cnn/classify", json={"result_id": rid}).get_json()
    assert cl["kind"] == "model_prediction" and "not a confirmed medical diagnosis" in cl["kind_note"]
    for name in ("cnn_mask", "cnn_overlay", "cnn_region", "cnn_probability"):
        assert c.get(seg["image_urls"][name]).status_code == 200
    cmp = seg["comparison"]
    assert 0 <= cmp["vs_reference"]["cnn"]["dice"] <= 1 and "pixel_accuracy" in cmp["vs_reference"]["cnn"]
    if cmp["classical_available"]:
        assert 0 <= cmp["agreement_cnn_vs_classical"]["dice"] <= 1 and "cnn_compare" in seg["image_urls"]
        assert "classical" in cmp["vs_reference"]
    # the CNN numbers equal an independent recomputation from the stored PNG
    from services.metrics import evaluate_masks
    stored = cv2.imread(str(res / rid / "cnn_mask.png"), cv2.IMREAD_GRAYSCALE)
    assert round(evaluate_masks(stored, ref)["dice"], 4) == cmp["vs_reference"]["cnn"]["dice"]
    # persisted in the result metadata and survives re-processing the classical pipeline
    assert {"segmentation", "classification"} <= set(c.get(f"/api/results/{rid}").get_json()["cnn"])
    c.post("/api/process", json={"result_id": rid, "params": {"selection": "largest"}})
    assert (res / rid / "cnn_mask.png").exists()
    meta = c.get(f"/api/results/{rid}").get_json()
    assert "cnn_mask" in meta["image_urls"] and "contour_overlay" in meta["image_urls"]


def test_cnn_evaluation_endpoint_serves_saved_test_metrics_only(trained):
    c, _ = _client(trained["models"])
    j = c.get("/api/cnn/evaluation").get_json()
    assert j["segmentation"]["available"] and j["classification"]["available"]
    assert j["segmentation"]["results"]["evaluated_on"] == "held-out test split" and j["segmentation"]["history"]["loss"]
    c2, _ = _client(trained["models"], seg=False, cls=False)
    j2 = c2.get("/api/cnn/evaluation").get_json()
    assert j2["segmentation"]["available"] is False and j2["segmentation"]["results"] is None and j2["segmentation"]["message"]


def test_classical_pipeline_unaffected_without_cnn(trained):
    c, _ = _client(trained["models"], seg=False, cls=False)
    img, ref = synthetic.circle()
    rid = upload(c, img, ref).get_json()["result_id"]
    r = c.post("/api/process", json={"result_id": rid, "params": {}})
    assert r.status_code == 200 and "contour_overlay" in r.get_json()["image_urls"]
    assert c.post("/api/evaluate", json={"result_id": rid}).get_json()["available"] is True


def test_threshold_tuning_end_to_end_uses_validation_only(trained):
    from cnn.classification import tune_threshold
    csv = Path(trained["images"]).parent / "ISIC_2016_Task3_GroundTruth.csv"
    mp = trained["models"] / "c.keras"
    sp = json.loads((trained["models"] / "c_split.json").read_text())
    t = tune_threshold.tune(mp, trained["images"], csv, target_recall=0.8, min_positives=3)
    import pytest as _p
    with _p.raises(Exception, match="too few"):
        tune_threshold.tune(mp, trained["images"], csv)                  # default guard: refuses 4 positives
    assert t["positive_class"] == "malignant" and 0.05 <= t["value"] <= 0.95
    assert f"{len(sp['val'])} images" in t["tuned_on"] and "validation" in t["tuned_on"]
    ev = cls_eval.evaluate_model(mp, trained["images"], csv)
    assert ev["decision_threshold"]["value"] == t["value"] and "at_default_threshold" in ev
    assert ev["n_test"] == len(sp["test"]) and ev["decision_rule"].startswith("predict 'malignant'")
    assert (trained["models"] / "c_threshold.json").exists()
    # the predictor applies the stored rule
    pred = cls_predict.ClassificationPredictor.load(mp)
    assert pred.threshold["value"] == t["value"]
    img, _ = make_case(1, 96, False)
    r = pred.predict(img)
    assert r["decision"]["positive_class"] == "malignant" and r["decision"]["threshold"] == t["value"]
    assert (r["predicted_class"] == "malignant") == (r["decision"]["positive_probability"] >= t["value"])
    # API + evaluation endpoint expose it
    c, _ = _client(trained["models"])
    j = c.post("/api/cnn/classify", data={"image": (io.BytesIO(png_bytes(img)), "x.png")}, content_type="multipart/form-data").get_json()
    assert j["prediction"]["decision"]["threshold"] == t["value"] and j["model"]["decision_threshold"] == t["value"]
    e = c.get("/api/cnn/evaluation").get_json()["classification"]
    assert e["threshold"]["chosen"]["threshold"] == t["value"] and e["results"]["at_default_threshold"]
    # --reset goes back to plain argmax
    tune_threshold.main(["--model-path", str(mp), "--images-dir", trained["images"], "--labels-csv", str(csv), "--reset"])
    assert cls_predict.ClassificationPredictor.load(mp).threshold is None
