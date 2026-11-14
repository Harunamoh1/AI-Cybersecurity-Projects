"""Integration tests of the REST API (in-memory database, real model)."""

import io
import os
import stat

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from sentinelai.api.main import create_app
from tests.conftest import make_settings, requires_model

pytestmark = requires_model


def upload(client, content: bytes, name="flows.csv", ctype="text/csv"):
    return client.post("/api/v1/analyses", files={"file": (name, content, ctype)})


def small_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode()


# ------------------------------------------------------------------ system


def test_health_and_security_headers(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["x-frame-options"] == "DENY"
    assert r.headers["cache-control"] == "no-store"


# --------------------------------------------------------------- prediction


def test_predict_json_flows(client, mixed_df):
    recs = mixed_df.drop(columns=["Label"]).head(5).to_dict("records")
    r = client.post("/api/v1/predict", json={"flows": recs, "explain": True})
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["predictions"]) == 5 and body["analysis_id"] is None
    p = body["predictions"][0]
    assert set(p["probabilities"]) >= {"BENIGN", "DDoS"} and 0 <= p["confidence"] <= 1
    assert p["explanation"]["contributions"]


def test_predict_accepts_canonical_names_and_numbers(client, detector, mixed_df):
    from sentinelai.features import canonicalize

    canon, _, _ = canonicalize(mixed_df.head(3))
    recs = canon.apply(pd.to_numeric, errors="coerce").to_dict("records")
    r = client.post("/api/v1/predict", json={"flows": recs})
    assert r.status_code == 200 and len(r.json()["predictions"]) == 3


def test_predict_save_creates_history(client, mixed_df):
    recs = mixed_df.drop(columns=["Label"]).head(4).to_dict("records")
    r = client.post("/api/v1/predict", json={"flows": recs, "save": True, "name": "../../etc/passwd"})
    aid = r.json()["analysis_id"]
    a = client.get(f"/api/v1/analyses/{aid}").json()
    assert a["source"] == "api" and a["total_flows"] == 4 and "/" not in a["filename"]


@pytest.mark.parametrize("payload,fragment", [
    ({"flows": []}, "flows"),
    ({"flows": [{"x" * 101: 1}]}, "100 characters"),
    ({"flows": [{"Flow Duration": "9" * 65}]}, "too long"),
    ({"flows": "not a list"}, "flows"),
])
def test_predict_validation_errors(client, payload, fragment):
    r = client.post("/api/v1/predict", json=payload)
    assert r.status_code == 422 and fragment in r.json()["detail"]


def test_predict_missing_features(client):
    r = client.post("/api/v1/predict", json={"flows": [{"Flow Duration": 1, "Unknown": 3}]})
    assert r.status_code == 422 and "Missing" in r.json()["detail"]


def test_predict_batch_limit(detector, mixed_df):
    with TestClient(create_app(settings=make_settings(max_batch=3), detector=detector)) as c:
        recs = mixed_df.drop(columns=["Label"]).head(4).to_dict("records")
        assert c.post("/api/v1/predict", json={"flows": recs}).status_code == 413


# ------------------------------------------------------------------- upload


def test_upload_sample_end_to_end(client, mixed_csv):
    r = upload(client, mixed_csv, name="mixed_traffic.csv")
    assert r.status_code == 201, r.text
    a = r.json()["analysis"]
    s = a["summary"]
    assert a["total_flows"] > 3000 and a["threats"] > 500
    assert s["ground_truth"]["accuracy"] > 0.95
    assert s["columns"]["label_column"] == "Label"
    assert set(s["class_counts"]) == {"BENIGN", "DoS", "DDoS", "PortScan", "BruteForce", "WebAttack", "Bot", "Infiltration"}
    top = r.json()["top_threats"]
    assert top and top[0]["predicted"] != "BENIGN" and top[0]["reasons"]


def test_upload_cic2018_sample_warns_about_shift(client):
    from tests.conftest import SAMPLES_DIR

    path = SAMPLES_DIR / "cicids2018_cross_dataset.csv"
    if not path.exists():
        pytest.skip("2018 sample not generated")
    r = upload(client, path.read_bytes())
    assert r.status_code == 201
    assert r.json()["analysis"]["summary"]["distribution_shift"]["level"] == "high"


def test_upload_semicolon_delimited_with_bom(client, mixed_df):
    data = b"\xef\xbb\xbf" + mixed_df.head(20).to_csv(index=False, sep=";").encode()
    r = upload(client, data)
    assert r.status_code == 201 and r.json()["analysis"]["total_flows"] == 20


def test_upload_missing_values_are_imputed_and_counted(client, mixed_df):
    df = mixed_df.head(30).copy()
    df.loc[:4, "Flow Bytes/s"] = ""
    df.loc[5, "Flow Packets/s"] = "Infinity"
    r = upload(client, small_csv(df))
    assert r.status_code == 201
    assert r.json()["analysis"]["summary"]["flows_with_imputed_values"] >= 6


def test_upload_unexpected_columns_are_reported(client, mixed_df):
    df = mixed_df.head(10).assign(Comment="hello", **{"Source IP": "192.168.0.1", "Timestamp": "x"})
    cols = upload(client, small_csv(df)).json()["analysis"]["summary"]["columns"]
    assert cols["unexpected_columns"] == ["Comment"]
    assert set(cols["identifiers_dropped"]) == {"Source IP", "Timestamp"}


@pytest.mark.parametrize("content,name,ctype,status,fragment", [
    (b"a,b\n1,2\n", "flows.exe", "text/csv", 415, ".csv"),
    (b"a,b\n1,2\n", "flows.csv", "application/x-msdownload", 415, "content type"),
    (b"\x89PNG\r\n\x1a\n\x00\x00\x00", "flows.csv", "text/csv", 415, "binary"),
    (b"", "flows.csv", "text/csv", 400, "empty"),
    (b"Flow Duration,Total Fwd Packets\n", "flows.csv", "text/csv", 400, "no flow records"),
    (b'a,b\n1,"unterminated\n2,3\n', "flows.csv", "text/csv", 400, "parse"),
    (b"a,b,c\n1,2,3\n4,5,6,7,8\n", "flows.csv", "text/csv", 400, "parse"),
    (b"foo,bar\n1,2\n", "flows.csv", "text/csv", 422, "Missing"),
])
def test_upload_rejections(client, content, name, ctype, status, fragment):
    r = upload(client, content, name=name, ctype=ctype)
    assert r.status_code == status, r.text
    assert fragment.lower() in r.json()["detail"].lower()


def test_upload_size_and_row_limits(detector, mixed_csv):
    with TestClient(create_app(settings=make_settings(max_upload_mb=0.05), detector=detector)) as c:
        assert upload(c, mixed_csv).status_code == 413
    with TestClient(create_app(settings=make_settings(max_rows=10), detector=detector)) as c:
        r = upload(c, mixed_csv)
        assert r.status_code == 413 and "rows" in r.json()["detail"]


def test_upload_column_limit(detector):
    with TestClient(create_app(settings=make_settings(max_columns=5), detector=detector)) as c:
        r = upload(c, ",".join(f"c{i}" for i in range(10)).encode() + b"\n" + b",".join(b"1" for _ in range(10)))
        assert r.status_code == 400 and "columns" in r.json()["detail"]


def test_upload_formula_cells_are_treated_as_data(client, mixed_df):
    df = mixed_df.head(5).copy()
    df.loc[0, "Flow Duration"] = "=cmd|' /C calc'!A0"
    r = upload(client, small_csv(df))
    assert r.status_code == 201  # coerced to NaN -> imputed, never evaluated
    assert r.json()["analysis"]["summary"]["flows_with_imputed_values"] >= 1


# ------------------------------------------------------- history & threats


def test_history_flows_explanations_and_deletion(client, mixed_csv):
    aid = upload(client, mixed_csv).json()["analysis"]["id"]
    page = client.get("/api/v1/analyses").json()
    assert page["total"] == 1 and page["items"][0]["id"] == aid

    flows = client.get(f"/api/v1/analyses/{aid}/flows", params={"cls": "DDoS", "limit": 5}).json()
    assert flows["total"] > 0 and all(f["predicted"] == "DDoS" for f in flows["items"])
    sev = client.get(f"/api/v1/analyses/{aid}/flows", params={"threats_only": True, "order": "severity"}).json()
    assert sev["items"][0]["severity"] == "critical"

    idx = flows["items"][0]["idx"]
    d = client.get(f"/api/v1/analyses/{aid}/flows/{idx}").json()
    assert d["explanation"]["predicted_class"] == "DDoS" and len(d["features"]) > 40
    assert client.get(f"/api/v1/analyses/{aid}/flows/999999").status_code == 404

    threats = client.get("/api/v1/threats", params={"severity": "critical", "min_confidence": 0.5}).json()
    assert threats["total"] > 0 and all(t["severity"] == "critical" and t["confidence"] >= 0.5 for t in threats["items"])

    stats = client.get("/api/v1/stats").json()
    assert stats["total_flows"] > 3000 and stats["threats"] > 0 and stats["labelled_agreement"] > 0.95
    assert len(stats["timeline"]) == 14 and stats["timeline"][-1]["total"] > 0

    assert client.delete(f"/api/v1/analyses/{aid}").json() == {"deleted": True}
    assert client.get(f"/api/v1/analyses/{aid}").status_code == 404
    assert client.get("/api/v1/threats").json()["total"] == 0  # flows cascade-deleted
    assert client.delete(f"/api/v1/analyses/{aid}").status_code == 404


def test_clear_history(client, mixed_df):
    upload(client, small_csv(mixed_df.head(10)))
    upload(client, small_csv(mixed_df.head(10)))
    assert client.delete("/api/v1/analyses").json() == {"deleted": 2}


@pytest.mark.parametrize("path", [
    "/api/v1/threats?severity=extreme", "/api/v1/threats?cls=Malware", "/api/v1/threats?limit=0",
    "/api/v1/analyses?limit=1000", "/api/v1/stats?days=1000",
])
def test_query_validation(client, path):
    assert client.get(path).status_code == 422


def test_retention_prunes_oldest(detector, mixed_df):
    with TestClient(create_app(settings=make_settings(retention_analyses=2), detector=detector)) as c:
        ids = [upload(c, small_csv(mixed_df.head(5))).json()["analysis"]["id"] for _ in range(3)]
        listed = [a["id"] for a in c.get("/api/v1/analyses").json()["items"]]
        assert ids[0] not in listed and set(ids[1:]) == set(listed)


def test_database_file_is_private(detector, tmp_path, mixed_df):
    db = tmp_path / "sub" / "sentinel.db"
    with TestClient(create_app(settings=make_settings(database_path=str(db)), detector=detector)) as c:
        upload(c, small_csv(mixed_df.head(5)))
    assert stat.S_IMODE(os.stat(db).st_mode) == 0o600


def test_ip_addresses_are_never_stored(detector, tmp_path, mixed_df):
    db = tmp_path / "s.db"
    df = mixed_df.head(5).assign(**{"Source IP": "203.0.113.77", "Destination IP": "198.51.100.9"})
    with TestClient(create_app(settings=make_settings(database_path=str(db)), detector=detector)) as c:
        upload(c, small_csv(df))
    raw = db.read_bytes() + (db.with_suffix(".db-wal").read_bytes() if db.with_suffix(".db-wal").exists() else b"")
    assert b"203.0.113.77" not in raw and b"198.51.100.9" not in raw


# ------------------------------------------------------------------ samples


def test_samples_list_download_and_analyse(client):
    samples = client.get("/api/v1/samples").json()
    names = {s["name"] for s in samples}
    assert "mixed_traffic.csv" in names and all("path" not in s for s in samples)
    r = client.get("/api/v1/samples/mixed_traffic.csv")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
    assert len(pd.read_csv(io.BytesIO(r.content))) > 3000
    r = client.post("/api/v1/samples/benign_monday.csv/analyze")
    assert r.status_code == 201 and r.json()["analysis"]["source"] == "sample"


@pytest.mark.parametrize("name", ["../models/model_meta.json", "..%2F..%2Fetc%2Fpasswd", "%2e%2e%2fREADME.md",
                                  "manifest.json", "nope.csv"])
def test_sample_path_traversal_is_impossible(client, name):
    assert client.get(f"/api/v1/samples/{name}").status_code == 404


# ------------------------------------------------------------------- model


def test_model_endpoints(client):
    m = client.get("/api/v1/model").json()
    assert m["deployed_model"] == "xgboost" and m["feature_count"] == len(m["features"]["selected"])
    assert {"random_forest", "xgboost", "decision_tree", "logistic_regression", "isolation_forest"} <= set(m["models"])
    assert m["calibration"]["after"]["ece"] <= m["calibration"]["before"]["ece"] + 0.01
    e = client.get("/api/v1/model/explainability").json()
    assert e["importance"] and e["beeswarm"] and set(e["per_class"]) == {c["name"] for c in m["classes"]}
    f = client.get("/api/v1/model/features").json()
    assert len(f["required"]) == m["feature_count"] and f["required"][0]["cic2018"]


# --------------------------------------------------------------- security


def test_api_key_enforced_when_configured(detector):
    with TestClient(create_app(settings=make_settings(api_key="s3cret-key"), detector=detector)) as c:
        assert c.get("/health").status_code == 200  # liveness stays open
        assert c.get("/api/v1/stats").status_code == 401
        assert c.get("/api/v1/stats", headers={"X-API-Key": "wrong"}).status_code == 401
        assert c.get("/api/v1/stats", headers={"X-API-Key": "s3cret-key"}).status_code == 200


def test_rate_limit(detector, mixed_df):
    with TestClient(create_app(settings=make_settings(rate_limit_per_minute=2), detector=detector)) as c:
        recs = mixed_df.drop(columns=["Label"]).head(1).to_dict("records")
        codes = [c.post("/api/v1/predict", json={"flows": recs}).status_code for _ in range(3)]
        assert codes == [200, 200, 429]


def test_cors_only_allows_configured_origin(client):
    ok = client.options("/api/v1/stats", headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET"})
    bad = client.options("/api/v1/stats", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "GET"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:3000"
    assert "access-control-allow-origin" not in bad.headers
