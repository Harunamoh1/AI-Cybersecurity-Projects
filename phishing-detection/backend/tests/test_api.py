"""Integration tests for the FastAPI service (real model, in-memory SQLite)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from phishlens.api.config import Settings
from phishlens.api.main import create_app

PHISH = "http://paypal.com.account-verify.secure-login.xyz/webscr?cmd=_login"
LEGIT = "https://en.wikipedia.org/wiki/Phishing"


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model"]


def test_security_headers(client):
    r = client.get("/health")
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["x-frame-options"] == "DENY"


class TestScan:
    def test_phishing_url(self, client):
        r = client.post("/api/v1/scan", json={"url": PHISH})
        assert r.status_code == 200
        body = r.json()
        assert body["verdict"] == "phishing"
        assert body["risk_level"] in ("high", "critical")
        assert body["id"] and body["created_at"]
        assert body["contributions"] and body["indicators"]
        assert body["components"]["registered_domain"] == "secure-login.xyz"

    def test_legitimate_url(self, client):
        body = client.post("/api/v1/scan", json={"url": LEGIT}).json()
        assert body["verdict"] == "legitimate"
        assert body["risk_level"] in ("low", "medium")
        assert 0 <= body["risk_score"] <= 100

    def test_scheme_optional(self, client):
        assert client.post("/api/v1/scan", json={"url": "en.wikipedia.org/wiki/Phishing"}).status_code == 200

    def test_score_is_computed_not_constant(self, client):
        a = client.post("/api/v1/scan", json={"url": PHISH, "save": False}).json()["probability"]
        b = client.post("/api/v1/scan", json={"url": LEGIT, "save": False}).json()["probability"]
        assert a != b

    def test_save_false_is_not_stored(self, client):
        client.post("/api/v1/scan", json={"url": LEGIT, "save": False})
        assert client.get("/api/v1/history").json()["total"] == 0

    @pytest.mark.parametrize(
        "url",
        ["javascript:alert(1)", "ftp://files.example.com/x", "http://exa mple.com", "http://", "data:text/html,hi",
         "http://example.com:70000/"],
    )
    def test_malformed_url_returns_400(self, client, url):
        r = client.post("/api/v1/scan", json={"url": url})
        assert r.status_code == 400
        assert isinstance(r.json()["detail"], str)

    @pytest.mark.parametrize(
        "payload",
        [{}, {"url": ""}, {"url": "   "}, {"url": 123}, {"url": None}, {"url": ["a.com"]}, {"url": "a" * 2049}],
    )
    def test_invalid_payload_returns_422(self, client, payload):
        r = client.post("/api/v1/scan", json=payload)
        assert r.status_code == 422
        assert isinstance(r.json()["detail"], str)

    def test_non_json_body(self, client):
        r = client.post("/api/v1/scan", content=b"not json", headers={"content-type": "application/json"})
        assert r.status_code == 422


class TestBatch:
    def test_mixed_batch(self, client):
        r = client.post("/api/v1/scan/batch", json={"urls": [PHISH, LEGIT, "javascript:void(0)"]})
        assert r.status_code == 200
        body = r.json()
        assert len(body["results"]) == 2
        assert len(body["errors"]) == 1 and "scheme" in body["errors"][0]["error"]
        assert client.get("/api/v1/history").json()["total"] == 2

    def test_batch_limit(self, client):
        r = client.post("/api/v1/scan/batch", json={"urls": ["a.com"] * 101})
        assert r.status_code == 400

    def test_empty_batch(self, client):
        assert client.post("/api/v1/scan/batch", json={"urls": []}).status_code == 422


class TestHistory:
    def _seed(self, client):
        ids = [client.post("/api/v1/scan", json={"url": u}).json()["id"] for u in (PHISH, LEGIT, "https://github.com/torvalds/linux")]
        return ids

    def test_list_newest_first_and_paginate(self, client):
        ids = self._seed(client)
        page = client.get("/api/v1/history", params={"limit": 2}).json()
        assert page["total"] == 3 and len(page["items"]) == 2
        assert page["items"][0]["id"] == ids[-1]
        page2 = client.get("/api/v1/history", params={"limit": 2, "offset": 2}).json()
        assert len(page2["items"]) == 1

    def test_filter_and_search(self, client):
        self._seed(client)
        phish = client.get("/api/v1/history", params={"verdict": "phishing"}).json()
        assert all(i["verdict"] == "phishing" for i in phish["items"]) and phish["total"] >= 1
        found = client.get("/api/v1/history", params={"q": "wikipedia"}).json()
        assert found["total"] == 1
        # LIKE wildcards and quotes are treated literally
        assert client.get("/api/v1/history", params={"q": "%' OR 1=1 --"}).json()["total"] == 0
        assert client.get("/api/v1/history", params={"q": "wiki_"}).json()["total"] == 0  # '_' is not a wildcard
        assert client.get("/api/v1/history", params={"q": "cmd=_"}).json()["total"] == 1  # but matches literally

    def test_invalid_query_params(self, client):
        assert client.get("/api/v1/history", params={"limit": 0}).status_code == 422
        assert client.get("/api/v1/history", params={"limit": 1000}).status_code == 422
        assert client.get("/api/v1/history", params={"verdict": "maybe"}).status_code == 422

    def test_get_and_delete_item(self, client):
        ids = self._seed(client)
        item = client.get(f"/api/v1/history/{ids[0]}").json()
        assert item["url"] == PHISH and item["contributions"]
        assert client.delete(f"/api/v1/history/{ids[0]}").json() == {"deleted": True}
        assert client.get(f"/api/v1/history/{ids[0]}").status_code == 404
        assert client.delete(f"/api/v1/history/{ids[0]}").status_code == 404

    def test_clear(self, client):
        self._seed(client)
        assert client.delete("/api/v1/history").json() == {"deleted": 3}
        assert client.get("/api/v1/history").json()["total"] == 0


def test_stats(client):
    empty = client.get("/api/v1/stats").json()
    assert empty["total_scans"] == 0 and empty["detection_rate"] == 0
    for u in (PHISH, LEGIT):
        client.post("/api/v1/scan", json={"url": u})
    s = client.get("/api/v1/stats", params={"days": 7}).json()
    assert s["total_scans"] == 2 and s["phishing"] == 1 and s["legitimate"] == 1
    assert len(s["timeline"]) == 7 and s["timeline"][-1]["phishing"] == 1
    assert sum(s["by_risk_level"].values()) == 2
    assert sum(b["count"] for b in s["score_histogram"]) == 2
    assert any(i["kind"] == "brand" for i in s["top_indicators"])


def test_model_card_reports_real_metrics(client, model):
    card = client.get("/api/v1/model").json()
    assert card["name"] == model.model_name
    test = card["models"][card["name"]]["test_at_tuned_threshold"]
    cm = test["confusion_matrix"]
    n = cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"]
    assert n == card["splits"]["test"]["rows"]
    assert test["accuracy"] == pytest.approx((cm["tn"] + cm["tp"]) / n)
    assert test["precision"] == pytest.approx(cm["tp"] / (cm["tp"] + cm["fp"]))
    assert test["recall"] == pytest.approx(cm["tp"] / (cm["tp"] + cm["fn"]))


def test_rate_limit(model):
    app = create_app(settings=Settings(database_path=":memory:", rate_limit_per_minute=2), model=model)
    with TestClient(app) as c:
        codes = [c.post("/api/v1/scan", json={"url": LEGIT, "save": False}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_openapi_schema(client):
    schema = client.get("/openapi.json").json()
    assert "/api/v1/scan" in schema["paths"]
