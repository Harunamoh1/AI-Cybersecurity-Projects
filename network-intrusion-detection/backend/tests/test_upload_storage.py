import io

import numpy as np
import pytest

from sentinelai.api.storage import Store
from sentinelai.api.upload import UploadError, decode, parse_csv, read_limited, safe_filename, validate_upload_meta


@pytest.mark.parametrize("raw,expected", [
    ("flows.csv", "flows.csv"),
    ("../../etc/passwd.csv", "passwd.csv"),
    ("C:\\Users\\x\\day1.csv", "day1.csv"),
    ("<script>alert(1)</script>.csv", "script_.csv"),  # "/" in "</script>" splits the path
    ("<img onerror=x>.csv", "_img onerror_x_.csv"),
    ("", "upload.csv"),
    (None, "upload.csv"),
    ("a" * 300 + ".csv", "a" * 100),
])
def test_safe_filename(raw, expected):
    assert safe_filename(raw) == expected


def test_read_limited():
    assert read_limited(io.BytesIO(b"x" * 100), 100) == b"x" * 100
    with pytest.raises(UploadError) as e:
        read_limited(io.BytesIO(b"x" * 101), 100)
    assert e.value.status_code == 413


def test_validate_upload_meta():
    validate_upload_meta("A.CSV", "text/csv; charset=utf-8")
    with pytest.raises(UploadError):
        validate_upload_meta("a.csv.exe", "text/csv")


def test_decode_latin1_fallback_and_bom():
    assert decode("Label\nWeb Attack \x96 XSS".encode("latin-1")).startswith("Label")
    assert decode(b"\xef\xbb\xbfa,b\n").startswith("a,b")


def test_parse_csv_drops_repeated_header_rows():
    data = b"Dst Port,Flow Duration,Label\n80,10,Benign\nDst Port,Flow Duration,Label\n443,20,Benign\n"
    df = parse_csv(data, max_rows=10, max_columns=10)
    assert df["Flow Duration"].tolist() == ["10", "20"]


def test_parse_csv_strips_header_whitespace():
    df = parse_csv(b" Flow Duration , Label\n1,BENIGN\n", 10, 10)
    assert list(df.columns) == ["Flow Duration", "Label"]


def _flows(n, cls="BENIGN"):
    return [{"index": i, "predicted_class": cls, "confidence": 0.9, "attack_probability": 0.1, "severity": "none",
             "reasons": []} for i in range(n)]


def test_store_roundtrip_and_feature_blob():
    s = Store(":memory:")
    feats = np.arange(12, dtype=np.float32).reshape(3, 4)
    flows = _flows(2) + [{**_flows(1, "Bot")[0], "index": 7, "severity": "critical",
                         "reasons": [{"feature": "f", "value": 1.0, "impact": 0.5}]}]
    a = s.add_analysis(filename="x.csv", source="upload", summary={"k": 1}, flows=flows, features=feats,
                       rejected_rows=0, duration_ms=1.0, model_version="1")
    assert a["threats"] == 1 and a["summary"] == {"k": 1}
    f = s.get_flow(a["id"], 7)
    np.testing.assert_array_equal(f["features"], feats[2])
    assert f["reasons"][0]["feature"] == "f"
    total, items = s.list_flows(threats_only=True)
    assert total == 1 and items[0]["filename"] == "x.csv"


def test_store_parameterised_queries_resist_injection():
    s = Store(":memory:")
    s.add_analysis(filename="x.csv", source="upload", summary={}, flows=_flows(1), features=np.zeros((1, 2)),
                   rejected_rows=0, duration_ms=1, model_version="1")
    assert s.get_analysis("' OR '1'='1") is None
    assert s.list_flows(analysis_id="' OR 1=1 --")[0] == 0
    assert s.list_analyses(10, 0)[0] == 1
