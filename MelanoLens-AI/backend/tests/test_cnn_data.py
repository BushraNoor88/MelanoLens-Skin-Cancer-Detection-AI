"""Dataset logic: ID matching (never row order), CSV inspection, leakage-free splits. No TensorFlow needed."""
import numpy as np
import pytest

from cnn import utils
from cnn.classification import dataset as cd
from cnn.errors import DatasetError
from tests.cnn_fixtures import write_dataset


def test_image_mask_pairing_by_id_with_and_without_suffix(tmp_path):
    root, _ = write_dataset(tmp_path / "a", n=10, mask_suffix="_Segmentation")
    pairs, rep = utils.pair_images_masks(root / "images", root / "masks")
    assert len(pairs) == 10 and rep["n_images_without_mask"] == 0
    assert all(p[1].stem == p[0] and p[2].stem.startswith(p[0]) for p in pairs)
    root2, _ = write_dataset(tmp_path / "b", n=5, mask_suffix="")
    assert len(utils.pair_images_masks(root2 / "images", root2 / "masks")[0]) == 5


def test_pairing_reports_unmatched_files(tmp_path):
    root, _ = write_dataset(tmp_path, n=8)
    (root / "masks" / "ISIC_0000003_Segmentation.png").unlink()
    (root / "masks" / "ISIC_9999999_Segmentation.png").write_bytes((root / "masks" / "ISIC_0000001_Segmentation.png").read_bytes())
    pairs, rep = utils.pair_images_masks(root / "images", root / "masks")
    assert len(pairs) == 7 and rep["images_without_mask"] == ["ISIC_0000003"] and rep["masks_without_image"] == ["ISIC_9999999"]


def test_pairing_fails_clearly_when_nothing_matches(tmp_path):
    (tmp_path / "i").mkdir(); (tmp_path / "m").mkdir()
    with pytest.raises(DatasetError):
        utils.pair_images_masks(tmp_path / "i", tmp_path / "m")


@pytest.mark.parametrize("labels,header,expected", [("text", True, ["benign", "malignant"]),
                                                    ("text", False, ["benign", "malignant"]),
                                                    ("float", False, ["0", "1"]), ("float", True, ["0", "1"])])
def test_csv_inspection_variants_and_row_order_independence(tmp_path, labels, header, expected):
    root, truth = write_dataset(tmp_path, n=24, labels=labels, header=header, shuffle_rows=True)
    rep, samples, paths = cd.inspect_labels_csv(root / "ISIC_2016_Task3_GroundTruth.csv", root / "images")
    assert rep["has_header_row"] is header and rep["classes"] == expected and rep["n_labelled_matched"] == 24
    assert rep["label_coding"] == ("numeric_code" if labels == "float" else "text")
    for (id_, lab), path in zip(samples, paths):                   # label really belongs to the SAME image id
        assert path.stem == id_ and lab == (truth[id_] if labels == "text" else str(int(float(truth[id_]))))
    assert sum(rep["class_counts"].values()) == 24


def test_csv_ids_with_extensions_and_missing_images(tmp_path):
    root, _ = write_dataset(tmp_path, n=10, header=False)
    csvp = root / "ISIC_2016_Task3_GroundTruth.csv"
    csvp.write_text("ISIC_0000000.jpg,benign\nISIC_0000001,malignant\nISIC_5555555,benign\nISIC_0000002,benign\n")
    rep, samples, _ = cd.inspect_labels_csv(csvp, root / "images")
    assert [s[0] for s in samples] == ["ISIC_0000000", "ISIC_0000001", "ISIC_0000002"]
    assert rep["n_csv_ids_without_image_file"] == 1 and rep["n_images_without_label"] == 7


def test_csv_ambiguous_label_columns_are_not_guessed(tmp_path):
    root, _ = write_dataset(tmp_path, n=12)
    rows = ["image_id,melanoma,keratosis"] + [f"ISIC_{i:07d},{i % 2}.0,{(i // 2) % 2}.0" for i in range(12)]
    p = root / "multi.csv"; p.write_text("\n".join(rows))
    (root / "ISIC_2016_Task3_GroundTruth.csv").unlink()
    with pytest.raises(DatasetError, match="Several possible label columns"):
        cd.inspect_labels_csv(p, root / "images")
    assert cd.inspect_labels_csv(p, root / "images", label_column="melanoma")[0]["label_column"] == "melanoma"


def test_csv_with_no_matching_ids_fails(tmp_path):
    root, _ = write_dataset(tmp_path, n=6)
    p = root / "bad.csv"; p.write_text("ID_1,benign\nID_2,malignant\n")
    with pytest.raises(DatasetError, match="image IDs"):
        cd.inspect_labels_csv(p, root / "images")


def test_find_labels_csv_ambiguity(tmp_path):
    root, _ = write_dataset(tmp_path, n=4)
    assert cd.find_labels_csv([root]).name == "ISIC_2016_Task3_GroundTruth.csv"
    (root / "other.csv").write_text("x")
    with pytest.raises(DatasetError, match="Several CSV"):
        cd.find_labels_csv([root])


def test_split_is_disjoint_stratified_deterministic():
    labels = [0] * 80 + [1] * 20
    a = utils.split_indices(100, labels, 0.15, 0.15, seed=1)
    assert a == utils.split_indices(100, labels, 0.15, 0.15, seed=1) != utils.split_indices(100, labels, 0.15, 0.15, seed=2)
    allidx = a["train"] + a["val"] + a["test"]
    assert sorted(allidx) == list(range(100))                       # nothing lost, nothing duplicated
    for part in ("val", "test"):
        assert 3 <= sum(labels[i] for i in a[part]) <= 5            # ~20 % positives kept in every split
    utils.check_no_leakage(utils.split_ids([f"id{i}" for i in range(100)], a))


def test_leakage_check_detects_overlap():
    with pytest.raises(DatasetError, match="leakage"):
        utils.check_no_leakage({"train": ["a", "b"], "test": ["b"]})


def test_class_weights_favour_rare_class():
    from cnn.classification.dataset import class_weights
    w = class_weights(np.array([0] * 90 + [1] * 10), 2)
    assert w[1] > w[0] and abs(w[0] * 90 + w[1] * 10 - 100) < 1e-6


def test_mask_resize_keeps_binary_values():
    m = np.zeros((50, 70), np.uint8); m[10:30, 20:50] = 255
    r = utils.resize_mask(m, 32)
    assert set(np.unique(r)) <= {0, 255} and r.shape == (32, 32)
