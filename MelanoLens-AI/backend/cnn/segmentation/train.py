"""Train the U-Net on ISIC Task 1 (run OUTSIDE Flask):

    cd backend
    python -m cnn.segmentation.train --epochs 30 --img-size 128 --batch-size 8

Everything (paths, sizes, batch size, epochs) is configurable; see --help.
"""
import argparse
import sys
from pathlib import Path

from config import Config

from .. import utils
from . import dataset, evaluate as seg_eval, model as seg_model


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train the U-Net lesion segmentation model (CPU friendly).")
    ap.add_argument("--images-dir", default=str(Config.ISIC_IMAGES_DIR))
    ap.add_argument("--masks-dir", default=str(Config.ISIC_MASKS_DIR))
    ap.add_argument("--model-path", default=str(Config.SEG_MODEL_PATH))
    ap.add_argument("--img-size", type=int, default=128, help="square input size (128 is CPU friendly; 256 is better but slower)")
    ap.add_argument("--base-filters", type=int, default=16)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=8, help="early-stopping patience (epochs)")
    ap.add_argument("--max-samples", type=int, default=None, help="use only N pairs (quick smoke test)")
    ap.add_argument("--no-eval", action="store_true", help="skip the final test-set evaluation")
    a = ap.parse_args(argv)

    tf, keras = utils.require_tf()
    keras.utils.set_random_seed(a.seed)
    model_path = Path(a.model_path); model_path.parent.mkdir(parents=True, exist_ok=True)

    pairs, report = utils.pair_images_masks(a.images_dir, a.masks_dir)
    if a.max_samples:
        pairs = pairs[: a.max_samples]
    print(f"Matched {len(pairs)} image/mask pairs by ID "
          f"({report['n_images_without_mask']} images without mask, {report['n_masks_without_image']} masks without image).")
    idx = utils.split_indices(len(pairs), None, a.val_frac, a.test_frac, a.seed)
    ids = [p[0] for p in pairs]
    split = utils.split_ids(ids, idx)
    utils.check_no_leakage(split)
    print(f"Split (disjoint): train={len(idx['train'])} val={len(idx['val'])} test={len(idx['test'])}")
    utils.write_json(utils.sidecar(model_path, "split"), {**split, "seed": a.seed})

    images, masks, mismatch = dataset.load_arrays(pairs, a.img_size)
    if mismatch:
        print(f"Note: {mismatch} image/mask pairs had different native sizes; both were resized to {a.img_size}x{a.img_size}.")
    sub = lambda k: (images[idx[k]], masks[idx[k]])
    train_ds = dataset.make_tf_dataset(*sub("train"), a.batch_size, augment=True, seed=a.seed)
    val_ds = dataset.make_tf_dataset(*sub("val"), a.batch_size)

    model = seg_model.compile_model(seg_model.build_unet(a.img_size, a.base_filters, a.depth), a.lr)
    ckpt = model_path.parent / "checkpoints" / f"{model_path.stem}_best.keras"
    ckpt.parent.mkdir(exist_ok=True)
    callbacks = [keras.callbacks.ModelCheckpoint(str(ckpt), monitor="val_loss", save_best_only=True),
                 keras.callbacks.EarlyStopping(monitor="val_loss", patience=a.patience, restore_best_weights=True),
                 keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)]
    hist = model.fit(train_ds, validation_data=val_ds, epochs=a.epochs, callbacks=callbacks, verbose=2)

    model.save(model_path)                                   # best weights (restored by EarlyStopping)
    history = {k: [round(float(x), 5) for x in v] for k, v in hist.history.items()}
    utils.write_json(utils.sidecar(model_path, "history"), history)
    utils.write_json(utils.sidecar(model_path, "meta"), {
        "model_name": "U-Net", "task": "segmentation", "trained_at": utils.now_iso(), "img_size": a.img_size,
        "base_filters": a.base_filters, "depth": a.depth, "batch_size": a.batch_size, "lr": a.lr, "seed": a.seed,
        "epochs_run": len(history["loss"]), "best_val_loss": round(min(history["val_loss"]), 5),
        "input_scaling": "RGB / 255", "loss": "binary cross-entropy + (1 - Dice)",
        "augmentation": "flips, rot90, brightness/contrast (train only)",
        "split_sizes": {k: len(v) for k, v in idx.items()}, "n_pairs": len(pairs),
        "tensorflow": tf.__version__, "keras": keras.__version__})
    print(f"Saved model: {model_path.name}  (history, meta and split saved next to it)")

    if not a.no_eval:
        r = seg_eval.evaluate_model(model_path, a.images_dir, a.masks_dir)
        print("Held-out TEST metrics (original resolution):")
        for k, v in r["metrics"].items():
            print(f"  {k:15s} mean={v['mean']:.4f}" if v else f"  {k}: n/a")
    return 0


if __name__ == "__main__":
    sys.exit(main())
