"""Verify the ISIC dataset on disk BEFORE training:  cd backend && python -m cnn.inspect_dataset

Prints: how many images/masks match by ID, the labels CSV columns it detected, the real class names and counts.
"""
import argparse
import json
import sys
from pathlib import Path

from config import Config

from . import utils
from .classification import dataset as cd
from .errors import DatasetError


def main(argv=None):
    ap = argparse.ArgumentParser(description="Inspect the ISIC dataset (images, masks, labels CSV).")
    ap.add_argument("--images-dir", default=str(Config.ISIC_IMAGES_DIR))
    ap.add_argument("--masks-dir", default=str(Config.ISIC_MASKS_DIR))
    ap.add_argument("--labels-csv", default=str(Config.ISIC_LABELS_CSV) if Config.ISIC_LABELS_CSV else None)
    ap.add_argument("--label-column", default=None)
    ap.add_argument("--json", action="store_true", help="print the report as JSON")
    a = ap.parse_args(argv)
    report, ok = {}, True
    try:
        _, report["segmentation_pairs"] = utils.pair_images_masks(a.images_dir, a.masks_dir)
    except DatasetError as e:
        report["segmentation_pairs"] = {"error": str(e)}; ok = False
    try:
        csv_path = cd.find_labels_csv([Path(a.images_dir).parent, Config.ISIC_DIR], a.labels_csv)
        report["labels_csv"] = cd.inspect_labels_csv(csv_path, a.images_dir, a.label_column)[0]
    except DatasetError as e:
        report["labels_csv"] = {"error": str(e)}; ok = False
    if a.json:
        print(json.dumps(report, indent=2))
    else:
        s, c = report["segmentation_pairs"], report["labels_csv"]
        print("== Segmentation (images <-> masks, matched by ID) ==")
        print("  ERROR: " + s["error"] if "error" in s else
              f"  images: {s['n_images']}  masks: {s['n_masks']}  matched pairs: {s['n_pairs']}\n"
              f"  images without mask: {s['n_images_without_mask']}  masks without image: {s['n_masks_without_image']}\n"
              f"  example names: {s['example_image_name']}  <->  {s['example_mask_name']}")
        print("== Classification (labels CSV) ==")
        if "error" in c:
            print("  ERROR: " + c["error"])
        else:
            print(f"  file: {c['csv_file']}  header row: {c['has_header_row']}  columns: {c['columns']}")
            print(f"  image-ID column: {c['id_column']!r}   label column: {c['label_column']!r}   coding: {c['label_coding']}")
            print(f"  rows: {c['n_csv_rows']}  matched to image files: {c['n_labelled_matched']}  "
                  f"CSV IDs without image: {c['n_csv_ids_without_image_file']}  images without label: {c['n_images_without_label']}")
            print(f"  classes ({c['n_classes']}, {'binary' if c['is_binary'] else 'multiclass'}): {c['class_counts']}")
            for n in c["notes"]:
                print("  note:", n)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
