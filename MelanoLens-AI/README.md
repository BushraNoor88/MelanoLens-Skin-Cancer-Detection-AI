# MelanoLens AI
**See the Edges. Understand the Boundaries.**

> MelanoLens AI is an educational computer vision project for experimental skin lesion boundary enhancement and image analysis. It is not a medical diagnostic tool. Its outputs must not be used to diagnose skin cancer or make treatment decisions.

MelanoLens AI is a **hybrid computer-vision application**:

| Path | Pipeline | Where |
|---|---|---|
| Classical CV | image -> preprocessing -> Edge Booster -> Canny/Sobel/... -> boundary extraction | `backend/services/` |
| CNN segmentation | image -> **U-Net** -> predicted mask -> boundary overlay | `backend/cnn/segmentation/` |
| CNN classification | image -> **MobileNetV2** classifier -> predicted diagnosis class | `backend/cnn/classification/` |

Backend: validation, preprocessing, Edge Booster, edge detectors, boundary extraction, metrics, CNN modules, REST API.
Frontend: 6 pages in plain HTML/CSS/vanilla JS served by Flask. See **CNN modules** below for training and use.

## Quick start (Windows PowerShell)
```powershell
cd MelanoLens-AI
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
cd backend
pytest
python app.py          # http://127.0.0.1:5000/api/health
```
If PowerShell blocks activation: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

## API quick reference (Milestone 2)
| Method | Path | Purpose |
|---|---|---|
| GET | /api/health | status |
| GET | /api/defaults | default parameters and option lists |
| POST | /api/upload | multipart: `image` (required), `mask` (optional reference) -> `result_id` |
| POST | /api/process | JSON `{"result_id": "...", "params": {...}}` runs the full pipeline |
| GET | /api/results/<id> | metadata, image and download URLs |
| GET | /api/results/<id>/image/<type> | PNG (e.g. original, enhanced, edges, contour_overlay, mask, reference_mask, difference, edges_conv_canny, edges_enh_canny) |
| GET | /api/results/<id>/download/<type> | same PNG as a download |
| DELETE | /api/results/<id> | delete all stored files of a result |
| POST | /api/evaluate | JSON `{"result_id": "..."}` -> metrics table (or "Ground-truth mask unavailable") |
| GET | /api/evaluate/<id>/export?format=csv\|json | metrics export |
| GET | /api/evaluate/<id>/chart.png | Matplotlib Dice chart |
| GET | /api/cnn/status | which CNN models are trained / loadable (never exposes file paths) |
| POST | /api/cnn/segment | multipart `image` **or** JSON `{"result_id"}` -> U-Net mask, region, boundary overlay, probability map, comparison with the classical result |
| POST | /api/cnn/classify | same inputs -> predicted class, confidence, all class probabilities, model info |
| GET | /api/cnn/evaluation | saved held-out **test-set** metrics + training history of both CNNs |

CNN errors use the same JSON schema; a missing/invalid model returns HTTP 503 (`model_missing`, `model_invalid`, `tensorflow_missing`). The app never returns placeholder predictions.

Uploads and results are stored in `data/results/<result_id>/`. Delete that folder (or call DELETE) to remove them.

## Using the app (Milestone 3)
```powershell
cd backend
python app.py
```
Open **http://127.0.0.1:5000/** (not by double-clicking index.html: the pages call the Flask API on the same address).

Demo flow: Upload (image + optional reference mask) -> Process Image -> Edge Booster workspace (change sliders, Process again)
-> Boundary Analysis (zoom, before/after slider, candidate contours, downloads) -> Evaluation (conventional vs enhanced table, chart, CSV/JSON export).
Frontend files: `frontend/*.html`, `frontend/css/`, `frontend/js/` (app.js = shared helpers, viewer.js = zoom/compare viewer).

## Practical notes from testing on a real ISIC image
- **Image frames:** many dermoscopy images have a 1-2 px bright/dark frame. "Ignore image border" (default 5 px) overwrites the outer pixels with their inner neighbours *before* filtering, so the frame cannot become a contour.
- **Auto Canny thresholds** (default on): high = percentile of the gradient magnitude, low = 0.4 x high, computed separately for the conventional and the enhanced image so both get a comparable edge budget. Fixed thresholds on the enhanced image can flood the edge map into one giant blob.
- **Dermoscopy preset** button (workspace): larger blur (9), closing kernel 11, "closest to centre" selection. It is a starting point tuned on a single image (ISIC_0000000), not a validated setting.
- Soft, fuzzy lesion borders give ragged contours (low circularity). Without a reference mask no accuracy can be computed; judge visually or add a mask.

## CNN modules (U-Net segmentation + MobileNetV2 classification)

The CNNs are **trained separately** from Flask; Flask only loads the saved models. Nothing is trained when you upload an image.
If a model file is missing the CNN buttons are disabled and the API answers 503 with instructions.

### 1. Install
```powershell
pip install -r backend\requirements.txt      # includes TensorFlow (CPU is enough). Python 3.9-3.12.
```
### 2. Put the ISIC data in `data/isic/` and verify it
```
data/isic/images/*.jpg          (ISIC 2016 Task 1 images)
data/isic/masks/*.png           (Task 1 masks; names like ISIC_0000000.png or ISIC_0000000_Segmentation.png both work)
data/isic/<Task 3 ground-truth>.csv
```
```powershell
cd backend
python -m cnn.inspect_dataset
```
This prints the matched image/mask pairs, the **actual CSV columns it detected**, the label column, class names and counts. It never assumes column names or `benign`/`malignant`: IDs are matched to files by **ID, not row order**; the ID column is found by matching values to image files; if several columns could be the label you must pick one with `--label-column`.
Paths can be changed with `--images-dir`, `--masks-dir`, `--labels-csv` or the env vars `ISIC_IMAGES_DIR`, `ISIC_MASKS_DIR`, `ISIC_LABELS_CSV`, `SEG_MODEL_PATH`, `CLS_MODEL_PATH`, `MODELS_DIR`.

### 3. Train (run from `backend/`; CPU friendly, every size/batch/epoch is a flag)
```powershell
python -m cnn.segmentation.train      --img-size 128 --batch-size 8  --epochs 30
python -m cnn.classification.train    --img-size 224 --batch-size 16 --epochs 10 --fine-tune-epochs 5
```
Add `--max-samples 60 --epochs 2` for a quick smoke test; use `--help` for all options (e.g. `--weights none` if you are offline, `--alpha 0.5` for a smaller MobileNetV2).
Each script splits the data into **disjoint train / validation / test sets** (seeded; classification is stratified), trains on train, selects the best epoch on validation, saves the model, and finally evaluates **only on the held-out test split**. Output next to each model in `backend/models/`:
`*.keras` model, `*_meta.json` (settings, class names), `*_history.json` (training curves), `*_split.json` (which IDs were train/val/test), `*_eval.json` (test metrics), `checkpoints/`.
Re-run only the evaluation: `python -m cnn.segmentation.evaluate` / `python -m cnn.classification.evaluate`.

### 4. Use it
```powershell
python app.py        # http://127.0.0.1:5000/
```
Upload an image (optionally with a reference mask) -> **Process** in the Workspace -> open **Boundary Analysis** (or the "CNN Analysis" button) -> scroll to **CNN analysis** -> *Run CNN segmentation* / *Classify lesion*. The page then shows the original, CNN mask, lesion region, CNN boundary overlay, classical overlay, a combined classical-vs-CNN boundary image, agreement scores, and the classification with class probabilities. The **Evaluation** page shows each model's held-out test metrics (Dice, IoU, precision, recall, pixel accuracy; accuracy, precision, recall, F1, confusion matrix, ROC-AUC) and training curves.


### Improving malignant recall: decision threshold (no re-training)
By default the classifier says "malignant" when that probability is above 0.5. When malignant cases are rare (here 19 %), this misses many of them. A lower threshold catches more but flags more benign lesions. Choose it **on the validation split only** (never on the test split):
```powershell
cd backend
python -m cnn.classification.tune_threshold --target-recall 0.80     # or: --criterion f2
python -m cnn.classification.tune_threshold --reset                   # back to plain 0.5
```
It picks the highest threshold whose validation recall reaches the target, saves it in `skin_lesion_classifier_meta.json` (the full sweep in `..._threshold.json`), then evaluates **once** on the held-out test split and prints default vs tuned results. The app, the API and the Evaluation page use the saved threshold automatically. The validation split has few malignant cases (about 26), so the chosen value is noisy; treat it as a trade-off, not a guarantee. Re-training removes the threshold; tune again afterwards.


### Classical vs CNN on the whole test set
```powershell
cd backend
python -m cnn.segmentation.compare          # about 10-30 min on a CPU laptop (classical runs 10 variants per image)
```
Scores the classical pipeline and the U-Net on the **same held-out test images**, at reference-mask resolution, with the same metric code. The classical pipeline produces 10 masks per image (5 detectors x conventional/enhanced); the headline classical method is the one with the best mean Dice on the **validation** split (never the test split), and all 10 are reported. If the classical pipeline finds no contour the image is scored as an empty mask (Dice 0) and counted. Reported: mean Dice/IoU with bootstrap 95 % intervals, the paired per-image difference with a Wilcoxon signed-rank p-value, and how many images each method won. Output: `unet_skin_lesion_comparison.json`, shown on the Evaluation page. Options: `--classical-params '{"selection":"closest_to_center"}'` (pipeline settings), `--max-images 10` (quick test), `--combo canny:enhanced` (skips validation selection; not recommended). This shows which method did better on this dataset and split, not which is better in general.

### Design notes
- **U-Net**: 4-level encoder-decoder with skip connections (base 16 filters, ~0.5 M parameters at the default), sigmoid output, loss = binary cross-entropy + (1 - Dice). Images resized with area interpolation and scaled to [0,1]; masks resized with **nearest neighbour**; augmentation (flips, 90-degree rotations, brightness/contrast) is applied to the training set only, and geometry is applied identically to image and mask. At inference the probability map is upsampled to the original size, thresholded at 0.5, and cleaned (largest component, holes filled).
- **Classifier**: MobileNetV2 (ImageNet weights, downloaded once) + dropout + softmax head. MobileNetV2's input scaling is a layer inside the model, so training and inference preprocess identically. Class imbalance is handled with balanced class weights; then the top layers are fine-tuned at a low learning rate. Class names come from the CSV and are stored with the model; numeric label codes are shown as codes because the file does not say what they mean.
- **Metrics**: segmentation metrics reuse `services/metrics.py` (computed at original resolution); classification metrics are plain NumPy (macro averages count a never-predicted class as 0; ROC-AUC for binary uses the class named malignant/melanoma if present, otherwise the last class, and says so).
- **Important**: the CNN output is a model prediction, not a diagnosis. A CNN is not guaranteed to beat the classical pipeline; scores on the Analysis page for an image the CNN trained on are optimistic, so use the held-out test metrics for fair comparison.

### Tests
```powershell
cd backend
pytest                      # classical + CNN tests (CNN integration tests skip if TensorFlow is missing)
```
The CNN tests train tiny models on **synthetic** images (never ISIC) to verify the code: ID matching, CSV inspection, leak-free splits, metric formulas against hand-computed values, save/load, API behaviour, and that the classical pipeline is unaffected. They say nothing about accuracy on real data.
