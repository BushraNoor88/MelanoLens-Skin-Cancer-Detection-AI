"""ISIC Task 3 labels: INSPECT the CSV (never assume column names / label coding), match IDs to image files.

Nothing here assumes 'benign'/'malignant'. Class names are whatever the CSV really contains.
"""
import csv
from collections import Counter
from pathlib import Path

import numpy as np

from .. import utils
from ..errors import DatasetError


def find_labels_csv(search_dirs, explicit=None):
    """Use the explicit path if given; otherwise the single .csv found in the first folder that has one."""
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            raise DatasetError(f"Labels CSV not found: {p.name!r}.")
        return p
    dirs = [Path(d) for d in ([search_dirs] if isinstance(search_dirs, (str, Path)) else search_dirs)]
    for d in dirs:
        found = sorted(d.glob("*.csv")) if d.is_dir() else []
        if len(found) == 1:
            return found[0]
        if len(found) > 1:
            raise DatasetError("Several CSV files found: " + ", ".join(f.name for f in found) +
                               ". Choose one with --labels-csv or ISIC_LABELS_CSV.")
    raise DatasetError("No .csv file found next to the images or in the ISIC folder. Put the Task 3 ground-truth "
                       "CSV in data/isic/ or set ISIC_LABELS_CSV / --labels-csv.")


def _read_rows(path):
    text = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    rows = [[c.strip() for c in r] for r in csv.reader(text.splitlines(), dialect) if any(c.strip() for c in r)]
    return rows, dialect.delimiter


def _as_number(v):
    try:
        return float(v)
    except ValueError:
        return None


def _normalise_labels(values):
    """Float-looking integers ('1.0') become '1'. Text labels are kept exactly as written."""
    nums = [_as_number(v) for v in values]
    if all(n is not None and float(n).is_integer() for n in nums):
        return [str(int(n)) for n in nums], "numeric_code"
    return list(values), "text"


def inspect_labels_csv(csv_path, images_dir, label_column=None):
    """Return a verification report (and the parsed ids/labels). Raises DatasetError with advice on problems."""
    rows, delim = _read_rows(csv_path)
    if len(rows) < 2:
        raise DatasetError("The labels CSV has no data rows.")
    files = {utils.normalize_id(p.name): p for p in utils.list_image_files(images_dir)}
    lower = {k.lower(): k for k in files}
    ncols = max(len(r) for r in rows)

    def match(cell):
        k = utils.normalize_id(cell)
        return k if k in files else lower.get(k.lower())

    ratios = [sum(match(r[j]) is not None for r in rows[1:] if j < len(r)) / max(len(rows) - 1, 1) for j in range(ncols)]
    id_col = int(np.argmax(ratios))
    if ratios[id_col] < 0.5:
        raise DatasetError("No CSV column contains image IDs matching the image files "
                           f"(best column matched {ratios[id_col]:.0%}). Check ISIC_IMAGES_DIR and the CSV.")
    has_header = match(rows[0][id_col]) is None
    header = rows[0] if has_header else [f"column_{j}" for j in range(ncols)]
    data = rows[1:] if has_header else rows

    candidates = []
    for j in range(ncols):
        if j == id_col:
            continue
        vals = [r[j] for r in data if j < len(r) and r[j] != ""]
        if 2 <= len(set(vals)) <= 20:
            candidates.append(j)
    if label_column is not None:
        chosen = header.index(label_column) if label_column in header else (int(label_column) if str(label_column).isdigit() else None)
        if chosen is None or chosen >= ncols or chosen == id_col:
            raise DatasetError(f"--label-column {label_column!r} not found. Columns: {header}")
        lab_col = chosen
    elif len(candidates) == 1:
        lab_col = candidates[0]
    elif not candidates:
        raise DatasetError(f"No label column found among columns {header} (need 2-20 distinct values).")
    else:                                   # several candidates: never guess which one is the diagnosis
        raise DatasetError("Several possible label columns: " + ", ".join(repr(header[j]) for j in candidates) +
                           ". Choose one with --label-column.")

    ids_raw, labels_raw = [], []
    for r in data:
        if id_col < len(r) and lab_col < len(r) and r[lab_col] != "":
            ids_raw.append(r[id_col]); labels_raw.append(r[lab_col])
    labels_all, coding = _normalise_labels(labels_raw)

    seen, samples, conflicts, missing = {}, [], [], []
    for raw, lab in zip(ids_raw, labels_all):
        key = match(raw)
        if key is None:
            missing.append(raw); continue
        if key in seen:
            if seen[key] != lab:
                conflicts.append(key)
            continue
        seen[key] = lab; samples.append((key, lab))
    if conflicts:
        raise DatasetError(f"{len(conflicts)} image IDs appear with conflicting labels (e.g. {conflicts[0]}).")
    if not samples:
        raise DatasetError("No labelled image could be matched to an image file.")

    counts = Counter(l for _, l in samples)
    classes = sorted(counts, key=lambda c: (float(c) if coding == "numeric_code" else 0, c))
    report = {"csv_file": Path(csv_path).name, "delimiter": delim, "has_header_row": has_header,
              "columns": header, "id_column": header[id_col], "label_column": header[lab_col],
              "n_csv_rows": len(data), "n_labelled_matched": len(samples),
              "n_csv_ids_without_image_file": len(missing), "csv_ids_without_image_file_examples": missing[:5],
              "n_images_without_label": len(set(files) - {k for k, _ in samples}),
              "n_duplicate_rows": len(ids_raw) - len(samples) - len(missing),
              "label_coding": coding, "classes": classes, "class_counts": {c: counts[c] for c in classes},
              "is_binary": len(classes) == 2, "n_classes": len(classes),
              "notes": ([f"Labels are numeric codes ({', '.join(classes)}) exactly as written in the CSV; their meaning "
                         "is not stated in the file, so they are shown as codes."] if coding == "numeric_code" else [])}
    return report, samples, [files[k] for k, _ in samples]


def load_labeled_samples(images_dir, csv_path, label_column=None):
    """-> dict(ids, paths, labels (int indices), class_names, report). Matching is by ID, never by row order."""
    report, samples, paths = inspect_labels_csv(csv_path, images_dir, label_column)
    names = report["classes"]
    return {"ids": [k for k, _ in samples], "paths": paths,
            "labels": np.array([names.index(l) for _, l in samples], dtype=np.int64),
            "class_names": names, "report": report}


def load_arrays(paths, size):
    """uint8 RGB images resized (INTER_AREA) to size x size."""
    return np.stack([utils.resize_image(utils.read_rgb(p), size) for p in paths])


def class_weights(train_labels, n_classes):
    """Balanced weights n / (k * count_c): rare classes count more in the loss."""
    counts = np.bincount(train_labels, minlength=n_classes).astype(float)
    return {c: float(len(train_labels) / (n_classes * counts[c])) if counts[c] else 1.0 for c in range(n_classes)}


def make_tf_dataset(images, labels, batch_size, augment=False, seed=42):
    """Images stay raw 0-255 float; the model's own Rescaling layer applies MobileNetV2 scaling."""
    tf, _ = utils.require_tf()
    ds = tf.data.Dataset.from_tensor_slices((tf.constant(images), tf.constant(labels)))
    if augment:
        ds = ds.shuffle(len(images), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=tf.data.AUTOTUNE)

    def aug(x, y):
        x = tf.image.random_flip_left_right(x)
        x = tf.image.random_flip_up_down(x)
        x = tf.image.rot90(x, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
        x = tf.image.random_brightness(x, 25.0)
        x = tf.image.random_contrast(x, 0.85, 1.15)
        return tf.clip_by_value(x, 0.0, 255.0), y

    if augment:
        ds = ds.map(aug, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def pick_positive_class(class_names):
    """Binary 'positive' class (used for ROC-AUC and the decision threshold). Prefer a class literally named
    malignant/melanoma; otherwise the last class in sorted order. Returns (index, explanation)."""
    for i, c in enumerate(class_names):
        if str(c).lower() in ("malignant", "melanoma"):
            return i, "named in the CSV"
    return len(class_names) - 1, "last class in sorted order (CSV does not name a malignant class)"
