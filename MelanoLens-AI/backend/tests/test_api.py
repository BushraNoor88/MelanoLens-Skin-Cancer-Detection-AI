import io
import json

import cv2
import numpy as np

from app import create_app
from services import synthetic
from tests.helpers import cleanup, make_client, png_bytes, upload


def _ids(resp):
    return resp.get_json()["result_id"]


def test_health_ok():
    r = create_app().test_client().get("/api/health")
    assert r.status_code == 200 and r.get_json()["status"] == "ok"


def test_unknown_route_returns_json_error():
    r = create_app().test_client().get("/api/does-not-exist")
    assert r.status_code == 404 and r.get_json()["error"]["code"] == "not_found"


def test_upload_validation_errors():
    client, tmp = make_client()
    try:
        img, _ = synthetic.circle()
        assert client.post("/api/upload", data={}, content_type="multipart/form-data").get_json()["error"]["code"] == "missing_file"
        bad_ext = client.post("/api/upload", data={"image": (io.BytesIO(png_bytes(img)), "x.gif")},
                              content_type="multipart/form-data")
        assert bad_ext.status_code == 400 and bad_ext.get_json()["error"]["code"] == "invalid_file_type"
        corrupt = client.post("/api/upload", data={"image": (io.BytesIO(b"not an image"), "x.png")},
                              content_type="multipart/form-data")
        assert corrupt.status_code == 400 and corrupt.get_json()["error"]["code"] == "corrupted_image"
        mism = upload(client, img, mask=np.zeros((100, 300), np.uint8))
        assert mism.status_code == 400 and mism.get_json()["error"]["code"] == "dimension_mismatch"
    finally:
        cleanup(tmp)


def test_upload_accepts_rgb_gray_and_reports_file_info():
    client, tmp = make_client()
    try:
        img, _ = synthetic.circle()
        for arr in (img, img[:, :, 0]):
            r = upload(client, arr)
            body = r.get_json()
            assert r.status_code == 201 and body["file"]["width"] == 256 and body["file"]["name"] == "lesion.png"
            assert body["reference_mask_available"] is False
    finally:
        cleanup(tmp)


def test_full_workflow_with_reference_mask():
    client, tmp = make_client()
    try:
        img, ref = synthetic.circle()
        rid = _ids(upload(client, img, ref))
        r = client.post("/api/process", json={"result_id": rid, "params": {"detector": "canny", "mode": "enhanced"}})
        assert r.status_code == 200
        meta = r.get_json()
        assert meta["boundary"]["selected_index"] is not None
        assert meta["boundary"]["measurements"]["area_px"] > 10000
        for name in ("original", "preprocessed", "enhanced", "edges", "contour_overlay", "mask",
                     "reference_mask", "difference", "edges_conv_canny", "edges_enh_prewitt"):
            assert name in meta["images"], name
        # image + download endpoints return real PNG files
        img_resp = client.get(meta["image_urls"]["mask"])
        assert img_resp.status_code == 200 and img_resp.data[:4] == b"\x89PNG"
        dl = client.get(meta["download_urls"]["contour_overlay"])
        assert "attachment" in dl.headers["Content-Disposition"] and dl.data[:4] == b"\x89PNG"
        # evaluation uses real saved masks
        ev = client.post("/api/evaluate", json={"result_id": rid}).get_json()
        assert ev["available"] is True and len(ev["rows"]) == 10 and len(ev["comparison"]) == 5
        canny_enh = next(x for x in ev["rows"] if x["detector"] == "canny" and x["mode"] == "enhanced")
        assert canny_enh["dice"] is not None and canny_enh["dice"] > 0.9
        assert ev["rows"] == meta["evaluation"]["rows"]
        csv_resp = client.get(f"/api/evaluate/{rid}/export?format=csv")
        assert csv_resp.status_code == 200 and csv_resp.data.decode().startswith("detector,mode,mask_found,dice")
        assert json.loads(client.get(f"/api/evaluate/{rid}/export?format=json").data)["available"] is True
        assert client.get(f"/api/evaluate/{rid}/chart.png").data[:4] == b"\x89PNG"
        assert client.get(f"/api/evaluate/{rid}/export?format=xml").status_code == 400
    finally:
        cleanup(tmp)


def test_missing_reference_mask_is_reported_not_faked():
    client, tmp = make_client()
    try:
        img, _ = synthetic.circle()
        rid = _ids(upload(client, img))
        client.post("/api/process", json={"result_id": rid})
        ev = client.post("/api/evaluate", json={"result_id": rid}).get_json()
        assert ev == {"result_id": rid, "available": False, "message": "Ground-truth mask unavailable"}
        assert client.get(f"/api/evaluate/{rid}/export").status_code == 404
        meta = client.get(f"/api/results/{rid}").get_json()
        assert meta["evaluation"]["available"] is False and "reference_mask" not in meta["images"]
    finally:
        cleanup(tmp)


def test_evaluate_before_processing_is_409():
    client, tmp = make_client()
    try:
        img, _ = synthetic.circle()
        rid = _ids(upload(client, img))
        r = client.post("/api/evaluate", json={"result_id": rid})
        assert r.status_code == 409 and r.get_json()["error"]["code"] == "not_processed"
    finally:
        cleanup(tmp)


def test_process_parameter_validation():
    client, tmp = make_client()
    try:
        img, _ = synthetic.circle()
        rid = _ids(upload(client, img))
        bad = [{"canny_auto": False, "canny_low": 200, "canny_high": 100}, {"kernel_size": 4}, {"booster_alpha": 50},
               {"detector": "nope"}, {"mystery": 1}, {"clahe": "yes"}, {"canny_low": "a"},
               {"filter": "median", "kernel_size": 1}, {"candidate_index": 99}]
        for params in bad:
            r = client.post("/api/process", json={"result_id": rid, "params": params})
            assert r.status_code == 400, params
            assert r.get_json()["error"]["code"] == "invalid_parameter", params
        assert client.post("/api/process", data="not json", content_type="text/plain").status_code == 400
    finally:
        cleanup(tmp)


def test_unknown_ids_and_unsafe_image_types():
    client, tmp = make_client()
    try:
        img, _ = synthetic.circle()
        rid = _ids(upload(client, img))
        client.post("/api/process", json={"result_id": rid})
        fake = "0" * 32
        assert client.get(f"/api/results/{fake}").status_code == 404
        assert client.post("/api/process", json={"result_id": fake}).get_json()["error"]["code"] == "result_not_found"
        assert client.post("/api/process", json={"result_id": "../../etc"}).status_code == 404
        assert client.get(f"/api/results/{rid}/image/nothing").status_code == 404
        assert client.get(f"/api/results/{rid}/image/_source").status_code == 404     # internal file hidden
        assert client.get(f"/api/results/{rid}/image/..%2Fmeta").status_code == 404
    finally:
        cleanup(tmp)


def test_flat_image_gives_warning_and_no_mask():
    client, tmp = make_client()
    try:
        flat = np.full((128, 128, 3), 120, np.uint8)
        rid = _ids(upload(client, flat))
        meta = client.post("/api/process", json={"result_id": rid}).get_json()
        assert meta["boundary"]["selected_index"] is None and meta["boundary"]["message"]
        assert "mask" not in meta["images"] and meta["warnings"]
    finally:
        cleanup(tmp)


def test_delete_removes_result():
    client, tmp = make_client()
    try:
        img, _ = synthetic.circle()
        rid = _ids(upload(client, img))
        assert client.delete(f"/api/results/{rid}").get_json()["deleted"] is True
        assert client.get(f"/api/results/{rid}").status_code == 404
    finally:
        cleanup(tmp)


def test_reprocessing_with_candidate_index_and_defaults_endpoint():
    client, tmp = make_client()
    try:
        img, _ = synthetic.circle()
        rid = _ids(upload(client, img))
        r = client.post("/api/process", json={"result_id": rid, "params": {"candidate_index": 0, "mode": "conventional"}})
        assert r.status_code == 200 and r.get_json()["boundary"]["mode"] == "conventional"
        d = client.get("/api/defaults").get_json()
        assert d["defaults"]["detector"] == "canny" and "prewitt" in d["options"]["detector"]
    finally:
        cleanup(tmp)


def test_manual_canny_thresholds_ignored_while_auto_is_on():
    client, tmp = make_client()
    try:
        img, _ = synthetic.circle()
        rid = _ids(upload(client, img))
        ok = client.post("/api/process", json={"result_id": rid, "params": {"canny_auto": True, "canny_low": 200, "canny_high": 100}})
        assert ok.status_code == 200 and ok.get_json()["canny_thresholds_used"]["enhanced"]["auto"] is True
    finally:
        cleanup(tmp)
