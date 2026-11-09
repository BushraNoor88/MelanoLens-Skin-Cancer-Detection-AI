"""Turns masks + edge maps + a reference mask into a comparison table, CSV and chart."""
import csv
import io

import numpy as np

from .edge_detection import DETECTORS
from .metrics import edge_map_boundary_f1, evaluate_masks
from .params import MODES

SHORT = {"conventional": "conv", "enhanced": "enh"}
METRIC_KEYS = ("dice", "iou", "precision", "recall", "f1", "boundary_f1")
CSV_COLUMNS = ("detector", "mode", "mask_found") + METRIC_KEYS + ("edge_map_boundary_f1",)
NO_REFERENCE_MESSAGE = "Ground-truth mask unavailable"


def _r(x):
    return None if x is None else round(float(x), 4)


def evaluate_combos(ref_mask, masks, edges, tolerance_px=2):
    """One row per (detector, mode). masks[(mode, det)] may be None if no contour was found."""
    rows = []
    for det in DETECTORS:
        for mode in MODES:
            mask, edge = masks.get((mode, det)), edges[(mode, det)]
            row = {"detector": det, "mode": mode, "mask_found": mask is not None,
                   **{k: None for k in METRIC_KEYS},
                   "edge_map_boundary_f1": _r(edge_map_boundary_f1(edge, ref_mask, tolerance_px))}
            if mask is not None:
                m = evaluate_masks(mask, ref_mask, tolerance_px)
                row.update({k: _r(m[k]) for k in METRIC_KEYS})
            rows.append(row)
    return rows


def comparison_summary(rows):
    """Enhanced minus conventional per detector. Pure arithmetic, no verdicts."""
    by = {(r["detector"], r["mode"]): r for r in rows}
    out = []
    for det in DETECTORS:
        c, e = by[(det, "conventional")], by[(det, "enhanced")]
        item = {"detector": det}
        for key in ("dice", "iou", "edge_map_boundary_f1"):
            item[f"{key}_conventional"], item[f"{key}_enhanced"] = c[key], e[key]
            item[f"{key}_delta"] = None if c[key] is None or e[key] is None else round(e[key] - c[key], 4)
        out.append(item)
    return out


def evaluate_stored(store, rid, meta):
    """Recompute everything from the files saved on disk (real data, not cached numbers)."""
    from .results_service import ResultStateError
    if not meta.get("processed"):
        raise ResultStateError("This result has not been processed yet. Call /api/process first.")
    ref = store.read_gray(rid, "reference_mask")
    if ref is None:
        return {"result_id": rid, "available": False, "message": NO_REFERENCE_MESSAGE}
    tol = meta["params"]["boundary_tolerance_px"]
    masks, edges = {}, {}
    for det in DETECTORS:
        for mode in MODES:
            masks[(mode, det)] = store.read_gray(rid, f"mask_{SHORT[mode]}_{det}")
            edges[(mode, det)] = store.read_gray(rid, f"edges_{SHORT[mode]}_{det}")
    rows = evaluate_combos(ref, masks, edges, tol)
    return {"result_id": rid, "available": True, "tolerance_px": tol, "rows": rows,
            "comparison": comparison_summary(rows)}


def rows_to_csv(rows):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(CSV_COLUMNS)
    for r in rows:
        w.writerow(["" if r[c] is None else r[c] for c in CSV_COLUMNS])
    return buf.getvalue()


def make_chart_png(comparison):
    """Grouped bar chart of Dice: conventional vs Edge Booster, from real computed values."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [c["detector"].capitalize() for c in comparison]
    conv = [c["dice_conventional"] for c in comparison]
    enh = [c["dice_enhanced"] for c in comparison]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=110)
    for offset, vals, color, name in ((-0.2, conv, "#64748b", "Conventional"),
                                      (0.2, enh, "#0ea5a4", "Edge Booster")):
        bars = ax.bar(x + offset, [v or 0 for v in vals], 0.38, color=color, label=name)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, (v or 0) + 0.02, "n/a" if v is None else f"{v:.2f}",
                    ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Dice (candidate mask vs reference)")
    ax.set_title("Conventional vs Edge Booster (n/a = no candidate contour)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()
