"""Train the lesion classifier on ISIC Task 3 labels (run OUTSIDE Flask):

    cd backend
    python -m cnn.classification.train --epochs 10 --fine-tune-epochs 5

The CSV is inspected first (ID column, label column, classes, counts) and the report is printed.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from config import Config

from .. import utils
from . import dataset, evaluate as cls_eval, model as cls_model


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train the MobileNetV2 lesion classifier (CPU friendly).")
    ap.add_argument("--images-dir", default=str(Config.ISIC_IMAGES_DIR))
    ap.add_argument("--labels-csv", default=str(Config.ISIC_LABELS_CSV) if Config.ISIC_LABELS_CSV else None)
    ap.add_argument("--label-column", default=None, help="column name or index, if the CSV has several candidates")
    ap.add_argument("--model-path", default=str(Config.CLS_MODEL_PATH))
    ap.add_argument("--img-size", type=int, default=224, help="224 recommended; 160 is faster on CPU")
    ap.add_argument("--alpha", type=float, default=1.0, help="MobileNetV2 width multiplier (0.35/0.5/0.75/1.0)")
    ap.add_argument("--weights", default="imagenet", help="'imagenet' (needs internet once) or 'none'")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=10, help="epochs with the base network frozen")
    ap.add_argument("--fine-tune-epochs", type=int, default=5, help="extra epochs with the top layers unfrozen (0 = skip)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--max-samples", type=int, default=None, help="use only N labelled images (smoke test)")
    ap.add_argument("--no-eval", action="store_true")
    a = ap.parse_args(argv)

    tf, keras = utils.require_tf()
    keras.utils.set_random_seed(a.seed)
    model_path = Path(a.model_path); model_path.parent.mkdir(parents=True, exist_ok=True)

    csv_path = dataset.find_labels_csv([Path(a.images_dir).parent, Config.ISIC_DIR], a.labels_csv)
    data = dataset.load_labeled_samples(a.images_dir, csv_path, a.label_column)
    rep = data["report"]
    print(f"CSV: {rep['csv_file']}  id column: {rep['id_column']!r}  label column: {rep['label_column']!r}  header row: {rep['has_header_row']}")
    print(f"Classes ({rep['label_coding']}): {rep['class_counts']}")
    for n in rep["notes"]:
        print("Note:", n)
    ids, paths, labels = data["ids"], data["paths"], data["labels"]
    if a.max_samples:
        keep = np.random.default_rng(a.seed).permutation(len(ids))[: a.max_samples]
        ids, paths, labels = [ids[i] for i in keep], [paths[i] for i in keep], labels[keep]
    idx = utils.split_indices(len(ids), labels.tolist(), a.val_frac, a.test_frac, a.seed)
    split = utils.split_ids(ids, idx)
    utils.check_no_leakage(split)
    print(f"Stratified split (disjoint): train={len(idx['train'])} val={len(idx['val'])} test={len(idx['test'])}")
    utils.write_json(utils.sidecar(model_path, "split"), {**split, "seed": a.seed})

    images = dataset.load_arrays(paths, a.img_size)
    sub = lambda k: (images[idx[k]], labels[idx[k]])
    train_ds = dataset.make_tf_dataset(*sub("train"), a.batch_size, augment=True, seed=a.seed)
    val_ds = dataset.make_tf_dataset(*sub("val"), a.batch_size)
    cw = dataset.class_weights(labels[idx["train"]], len(data["class_names"]))
    print("Class weights (imbalance handling):", {data["class_names"][c]: round(w, 3) for c, w in cw.items()})

    model, base = cls_model.build_classifier(len(data["class_names"]), a.img_size, a.alpha, a.weights)
    cls_model.compile_model(model, a.lr)
    ckpt = model_path.parent / "checkpoints" / f"{model_path.stem}_best.keras"; ckpt.parent.mkdir(exist_ok=True)
    cbs = lambda: [keras.callbacks.ModelCheckpoint(str(ckpt), monitor="val_loss", save_best_only=True),
                   keras.callbacks.EarlyStopping(monitor="val_loss", patience=a.patience, restore_best_weights=True)]
    history = {}
    def collect(h):
        for k, v in h.history.items():
            history.setdefault(k, []).extend(round(float(x), 5) for x in v)
    collect(model.fit(train_ds, validation_data=val_ds, epochs=a.epochs, class_weight=cw, callbacks=cbs(), verbose=2))
    if a.fine_tune_epochs > 0:
        print("Fine-tuning the top layers of MobileNetV2 with a low learning rate...")
        cls_model.unfreeze_top(base); cls_model.compile_model(model, a.lr / 50)
        collect(model.fit(train_ds, validation_data=val_ds, epochs=a.fine_tune_epochs, class_weight=cw, callbacks=cbs(), verbose=2))

    model.save(model_path)
    utils.write_json(utils.sidecar(model_path, "history"), history)
    utils.write_json(utils.sidecar(model_path, "meta"), {
        "model_name": "MobileNetV2 (transfer learning)", "task": "classification", "trained_at": utils.now_iso(),
        "class_names": data["class_names"], "label_column": rep["label_column"], "label_coding": rep["label_coding"],
        "class_counts": rep["class_counts"], "img_size": a.img_size, "alpha": a.alpha,
        "weights": None if a.weights == "none" else a.weights, "batch_size": a.batch_size, "seed": a.seed,
        "input_scaling": "raw RGB 0-255; MobileNetV2 scaling is a layer inside the model",
        "class_weights": {data["class_names"][c]: round(w, 4) for c, w in cw.items()},
        "augmentation": "flips, rot90, brightness/contrast (train only)",
        "split_sizes": {k: len(v) for k, v in idx.items()}, "tensorflow": tf.__version__, "keras": keras.__version__})
    print(f"Saved model: {model_path.name}  (history, meta and split saved next to it)")
    if not a.no_eval:
        print("Held-out TEST metrics:")
        cls_eval.print_summary(cls_eval.evaluate_model(model_path, a.images_dir, csv_path, a.label_column))
    return 0


if __name__ == "__main__":
    sys.exit(main())
