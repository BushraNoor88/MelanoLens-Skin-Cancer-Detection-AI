"""Classical CV vs U-Net on the SAME held-out test images (fair, paired comparison).

    cd backend
    python -m cnn.segmentation.compare

Protocol (written to the output JSON too):
 * Test images = the U-Net's held-out test split (never used for training or model selection).
 * Classical = the app's pipeline with its default parameters (or --classical-params). It produces 10 masks per
   image (5 detectors x conventional/enhanced). The "headline" classical method is the combination with the best
   mean Dice on the VALIDATION split. The test split is never used to choose it. All 10 are still reported.
 * Both methods are scored at the reference-mask resolution with the same metric code. If the classical pipeline
   finds no candidate contour, its mask is empty (Dice 0) and it is counted in `n_no_mask`.
 * Paired statistics on per-image Dice/IoU: mean difference with a bootstrap 95 % interval, Wilcoxon signed-rank
   test, and wins / ties / losses (a tie = difference under 0.01).
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy import stats

from config import Config
from services.edge_detection import DETECTORS
from services.evaluation_service import SHORT
from services.metrics import evaluate_masks
from services.params import MODES, parse_process_params
from services.pipeline import run_pipeline

from .. import utils
from ..errors import DatasetError
from .predict import SegmentationPredictor

KEYS = ("dice", "iou", "precision", "recall", "pixel_accuracy")
TIE = 0.01


def score(mask, ref_bool):
    """Metrics of one predicted mask (None or empty = nothing predicted) against the reference, at reference size."""
    h, w = ref_bool.shape
    if mask is None:
        pred = np.zeros((h, w), np.uint8)
    else:
        m = (mask > 0).astype(np.uint8) * 255
        pred = m if m.shape == (h, w) else cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    r = evaluate_masks(pred, ref_bool.astype(np.uint8) * 255)
    total = r["tp"] + r["fp"] + r["fn"] + r["tn"]
    return {"dice": r["dice"], "iou": r["iou"], "precision": r["precision"], "recall": r["recall"],
            "pixel_accuracy": (r["tp"] + r["tn"]) / total, "found": mask is not None and bool((mask > 0).any())}


def classical_scores(bgr, ref_bool, params):
    out = run_pipeline(bgr, params, None)["images"]
    return {f"{det}:{mode}": score(out.get(f"mask_{SHORT[mode]}_{det}"), ref_bool) for det in DETECTORS for mode in MODES}


def bootstrap_ci(values, n_boot=10000, seed=0, alpha=0.05):
    v = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def summarize(values, n_boot=10000, seed=0):
    v = np.asarray([x for x in values if x is not None], float)
    if v.size == 0:
        return None
    lo, hi = bootstrap_ci(v, n_boot, seed)
    r = lambda x: round(float(x), 4)
    return {"mean": r(v.mean()), "std": r(v.std()), "median": r(np.median(v)), "ci95": [r(lo), r(hi)], "n": int(v.size)}


def paired(cnn, classical, n_boot=10000, seed=0):
    """CNN minus classical, per image."""
    d = np.asarray(cnn, float) - np.asarray(classical, float)
    lo, hi = bootstrap_ci(d, n_boot, seed)
    try:
        p = float(stats.wilcoxon(d, zero_method="wilcox").pvalue) if np.any(d != 0) else None
    except ValueError:
        p = None
    r = lambda x: round(float(x), 4)
    return {"mean_difference": r(d.mean()), "ci95": [r(lo), r(hi)], "wilcoxon_p": None if p is None else float(f"{p:.3g}"),
            "cnn_better": int((d > TIE).sum()), "ties": int((np.abs(d) <= TIE).sum()), "classical_better": int((d < -TIE).sum()),
            "n": int(d.size)}


def compare(model_path, images_dir, masks_dir, classical_params=None, combo=None, max_images=None, n_boot=10000, seed=0, out_path=None):
    model_path = Path(model_path)
    split = utils.read_json(utils.sidecar(model_path, "split"))
    if not split:
        raise DatasetError("No split file next to the segmentation model; train it first.")
    utils.check_no_leakage({k: split[k] for k in ("train", "val", "test")})
    pairs, _ = utils.pair_images_masks(images_dir, masks_dir)
    by_id = {p[0]: p for p in pairs}
    val_ids = [i for i in split["val"] if i in by_id][:max_images]
    test_ids = [i for i in split["test"] if i in by_id][:max_images]
    if not test_ids:
        raise DatasetError("None of the test-split images were found on disk.")
    params = parse_process_params(classical_params or {})
    names = [f"{d}:{m}" for d in DETECTORS for m in MODES]

    selection = {"selected_on": "validation split", "n_validation": 0, "validation_mean_dice": None}
    if combo:
        if combo not in names:
            raise DatasetError(f"--combo must be one of {names}")
        chosen = combo
        selection = {"selected_on": "user choice (--combo); not selected from data", "n_validation": 0, "validation_mean_dice": None}
    else:
        print(f"Selecting the classical method on the validation split ({len(val_ids)} images)...")
        val = {n: [] for n in names}
        for k, i in enumerate(val_ids, 1):
            _id, ip, mp = by_id[i]
            rgb = utils.read_rgb(ip); ref = utils.read_mask(mp) >= 128
            for n, s in classical_scores(rgb[..., ::-1], ref, params).items():
                val[n].append(s["dice"])
            if k % 10 == 0 or k == len(val_ids):
                print(f"  validation: {k}/{len(val_ids)} images", flush=True)
        means = {n: float(np.mean(v)) for n, v in val.items()}
        chosen = max(names, key=lambda n: means[n])
        selection.update({"n_validation": len(val_ids), "validation_mean_dice": {n: round(m, 4) for n, m in means.items()}})
    selection["chosen"] = chosen

    print(f"Scoring on the held-out test split ({len(test_ids)} images); classical headline = {chosen}")
    predictor = SegmentationPredictor.load(model_path)
    per_image, cls_all = [], {n: [] for n in names}
    for k, i in enumerate(test_ids, 1):
        _id, ip, mp = by_id[i]
        rgb = utils.read_rgb(ip); ref = utils.read_mask(mp) >= 128
        cnn = score(predictor.predict(rgb[..., ::-1])["mask"].astype(np.uint8) * 255, ref)
        cls = classical_scores(rgb[..., ::-1], ref, params)
        for n in names:
            cls_all[n].append(cls[n])
        per_image.append({"id": i, "cnn_dice": round(cnn["dice"], 4), "classical_dice": round(cls[chosen]["dice"], 4),
                          "cnn_iou": round(cnn["iou"], 4), "classical_iou": round(cls[chosen]["iou"], 4),
                          "classical_found": cls[chosen]["found"], "cnn_found": cnn["found"],
                          "_cnn": cnn, "_cls": cls[chosen]})
        if k % 10 == 0 or k == len(test_ids):
            print(f"  test: {k}/{len(test_ids)} images", flush=True)

    col = lambda rows, key: [r[key] for r in rows]
    headline = {"cnn": {m: summarize(col([r["_cnn"] for r in per_image], m), n_boot, seed) for m in KEYS},
                "classical": {m: summarize(col([r["_cls"] for r in per_image], m), n_boot, seed) for m in KEYS}}
    result = {
        "task": "segmentation_comparison", "evaluated_on": "held-out test split", "evaluated_at": utils.now_iso(),
        "n_test": len(test_ids), "split_sizes": {k: len(v) for k, v in split.items() if k in ("train", "val", "test")},
        "classical": {"headline_method": chosen, "parameters": "app defaults" if not classical_params else classical_params,
                      "selection": selection, "n_no_mask": int(sum(not r["classical_found"] for r in per_image)),
                      "n_no_mask_note": "images where the classical pipeline found no candidate contour (scored as an empty mask, Dice 0)"},
        "cnn": {"model": "U-Net", "n_empty_prediction": int(sum(not r["cnn_found"] for r in per_image))},
        "headline": headline,
        "paired": {"dice": paired(col(per_image, "cnn_dice"), col(per_image, "classical_dice"), n_boot, seed),
                   "iou": paired(col(per_image, "cnn_iou"), col(per_image, "classical_iou"), n_boot, seed)},
        "classical_all_methods": [{"method": n, "dice": summarize([s["dice"] for s in cls_all[n]], n_boot, seed),
                                   "iou": summarize([s["iou"] for s in cls_all[n]], n_boot, seed),
                                   "n_no_mask": int(sum(not s["found"] for s in cls_all[n]))} for n in names],
        "per_image": [{k: v for k, v in r.items() if not k.startswith("_")} for r in per_image],
        "protocol": ["Same held-out test images, same reference masks, same metric code and resolution for both methods.",
                     "The headline classical method was chosen on the validation split only.",
                     "Classical parameters are the app defaults (not tuned per image); results can differ with other settings.",
                     "Intervals are bootstrap 95% intervals over images (10 000 resamples); p is a Wilcoxon signed-rank test.",
                     "One dataset, one split: this shows which method did better here, not which is better in general."]}
    utils.write_json(out_path or utils.sidecar(model_path, "comparison"), result)
    return result


def print_summary(r):
    c, k, p = r["headline"]["classical"], r["headline"]["cnn"], r["paired"]["dice"]
    print(f"\nTest images: {r['n_test']}   classical headline method: {r['classical']['headline_method']} "
          f"(chosen on validation: {r['classical']['selection']['selected_on']})")
    print(f"{'':12s}{'Dice':>22s}{'IoU':>22s}")
    for name, m in (("Classical", c), ("U-Net", k)):
        print(f"{name:12s}{m['dice']['mean']:>8.3f}  [{m['dice']['ci95'][0]:.3f}, {m['dice']['ci95'][1]:.3f}]"
              f"{m['iou']['mean']:>10.3f}  [{m['iou']['ci95'][0]:.3f}, {m['iou']['ci95'][1]:.3f}]")
    print(f"Paired Dice difference (U-Net - classical): {p['mean_difference']:+.3f}  95% CI {p['ci95']}  Wilcoxon p={p['wilcoxon_p']}")
    print(f"Per image: U-Net better {p['cnn_better']}, tie {p['ties']}, classical better {p['classical_better']}")
    print(f"Classical found no contour on {r['classical']['n_no_mask']} images; U-Net predicted nothing on {r['cnn']['n_empty_prediction']}.")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Compare the classical pipeline with the U-Net on the held-out test split.")
    ap.add_argument("--model-path", default=str(Config.SEG_MODEL_PATH))
    ap.add_argument("--images-dir", default=str(Config.ISIC_IMAGES_DIR))
    ap.add_argument("--masks-dir", default=str(Config.ISIC_MASKS_DIR))
    ap.add_argument("--classical-params", default=None, help='JSON of pipeline parameters, e.g. \'{"selection":"closest_to_center"}\'')
    ap.add_argument("--combo", default=None, help="skip validation selection and use e.g. canny:enhanced (not recommended)")
    ap.add_argument("--max-images", type=int, default=None, help="use only the first N validation/test images (smoke test)")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    r = compare(a.model_path, a.images_dir, a.masks_dir, json.loads(a.classical_params) if a.classical_params else None,
                a.combo, a.max_images, a.bootstrap, a.seed)
    print_summary(r)
    print("\nSaved next to the model as *_comparison.json (shown on the Evaluation page).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
