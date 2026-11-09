"""SYNTHETIC ISIC-shaped fixture data for tests. NOT real data and never used for reported results.

Layout mimics the real dataset: images/ISIC_xxxxxxx.jpg, masks/ISIC_xxxxxxx_Segmentation.png, a labels CSV.
"cls1" lesions are darker and irregular, "cls0" lesions are lighter and round, so a tiny CNN can learn them.
"""
import csv
from pathlib import Path

import cv2
import numpy as np


def make_case(i, size=96, malignant=False):
    rng = np.random.default_rng(i)
    bg = np.array([170, 190, 215], np.float32) + rng.normal(0, 4, 3)           # skin-like BGR
    img = np.tile(bg, (size, size, 1)).astype(np.float32)
    mask = np.zeros((size, size), np.uint8)
    cx, cy = size // 2 + rng.integers(-6, 7), size // 2 + rng.integers(-6, 7)
    r = rng.integers(size // 5, size // 3)
    if malignant:
        pts = [(cx + int((r * rng.uniform(.6, 1.3)) * np.cos(a)), cy + int((r * rng.uniform(.6, 1.3)) * np.sin(a)))
               for a in np.linspace(0, 2 * np.pi, 9)[:-1]]
        cv2.fillPoly(mask, [np.array(pts, np.int32)], 255); col = (40, 50, 70)
    else:
        cv2.circle(mask, (cx, cy), r, 255, -1); col = (110, 130, 170)
    img[mask > 0] = np.array(col, np.float32)
    img += rng.normal(0, 5, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8), mask


def write_dataset(root, n=48, size=96, labels="text", header=True, mask_suffix="_Segmentation", shuffle_rows=True,
                  csv_name="ISIC_2016_Task3_GroundTruth.csv"):
    root = Path(root); (root / "images").mkdir(parents=True, exist_ok=True); (root / "masks").mkdir(exist_ok=True)
    rows = []
    for i in range(n):
        mal = i % 4 == 0                                                       # 25 % class 1 (imbalanced)
        img, mask = make_case(i, size, mal)
        name = f"ISIC_{i:07d}"
        cv2.imwrite(str(root / "images" / f"{name}.jpg"), img)
        cv2.imwrite(str(root / "masks" / f"{name}{mask_suffix}.png"), mask)
        lab = {"text": "malignant" if mal else "benign", "float": "1.0" if mal else "0.0"}[labels]
        rows.append([name, lab])
    if shuffle_rows:
        np.random.default_rng(0).shuffle(rows)                                 # row order != file order
    with open(root / csv_name, "w", newline="") as f:
        w = csv.writer(f)
        if header:
            w.writerow(["image_id", "diagnosis"])
        w.writerows(rows)
    truth = {r[0]: r[1] for r in rows}
    return root, truth
