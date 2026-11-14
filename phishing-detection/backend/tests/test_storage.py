"""Unit tests for the SQLite scan store."""

from __future__ import annotations

from phishlens.api.storage import ScanStore


def _result(url: str, verdict: str = "legitimate", score: float = 10.0) -> dict:
    return {
        "url": url, "verdict": verdict, "risk_level": "low" if verdict == "legitimate" else "critical",
        "risk_score": score, "probability": score / 100,
        "components": {"registered_domain": "example.com", "suffix": "com"}, "indicators": [],
    }


def test_roundtrip_and_persistence(tmp_path):
    db = str(tmp_path / "scans.db")
    store = ScanStore(db)
    rec = store.add(_result("https://example.com/a"))
    store.close()
    reopened = ScanStore(db)
    assert reopened.get(rec["id"])["url"] == "https://example.com/a"
    total, items = reopened.list(10, 0)
    assert total == 1 and items[0]["id"] == rec["id"]


def test_stats_empty_and_populated():
    store = ScanStore(":memory:")
    assert store.stats()["total_scans"] == 0
    store.add(_result("https://a.com", "phishing", 95))
    store.add(_result("https://b.com", "legitimate", 5))
    s = store.stats()
    assert s["phishing"] == 1 and s["average_risk_score"] == 50
    assert s["score_histogram"][9]["count"] == 1 and s["score_histogram"][0]["count"] == 1
