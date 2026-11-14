"""SQLite-backed scan history.

SQLite keeps the project dependency-free and is plenty for a single-node
deployment. All queries are parameterised.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from collections import Counter
from datetime import UTC, datetime, timedelta
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS scans (
    id            TEXT PRIMARY KEY,
    created_at    TEXT NOT NULL,
    url           TEXT NOT NULL,
    registered_domain TEXT,
    suffix        TEXT,
    verdict       TEXT NOT NULL,
    risk_level    TEXT NOT NULL,
    risk_score    REAL NOT NULL,
    probability   REAL NOT NULL,
    result_json   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_scans_created ON scans(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_scans_verdict ON scans(verdict);
"""


class ScanStore:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        if path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def add(self, result: dict[str, Any]) -> dict[str, Any]:
        scan_id = uuid.uuid4().hex
        created = datetime.now(UTC).isoformat(timespec="seconds")
        record = {**result, "id": scan_id, "created_at": created}
        with self._lock:
            self._conn.execute(
                "INSERT INTO scans (id, created_at, url, registered_domain, suffix, verdict, risk_level,"
                " risk_score, probability, result_json) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    scan_id, created, result["url"], result["components"]["registered_domain"],
                    result["components"]["suffix"], result["verdict"], result["risk_level"],
                    result["risk_score"], result["probability"], json.dumps(record),
                ),
            )
            self._conn.commit()
        return record

    def get(self, scan_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT result_json FROM scans WHERE id = ?", (scan_id,)).fetchone()
        return json.loads(row["result_json"]) if row else None

    def list(self, limit: int, offset: int, verdict: str | None = None, query: str | None = None) -> tuple[int, list[dict]]:
        where, params = [], []
        if verdict:
            where.append("verdict = ?")
            params.append(verdict)
        if query:
            where.append("url LIKE ? ESCAPE '\\'")
            escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            params.append(f"%{escaped}%")
        clause = (" WHERE " + " AND ".join(where)) if where else ""
        with self._lock:
            total = self._conn.execute(f"SELECT COUNT(*) FROM scans{clause}", params).fetchone()[0]  # noqa: S608
            rows = self._conn.execute(
                f"SELECT id, created_at, url, registered_domain, verdict, risk_level, risk_score"  # noqa: S608
                f" FROM scans{clause} ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?",
                [*params, limit, offset],
            ).fetchall()
        return total, [dict(r) for r in rows]

    def clear(self) -> int:
        with self._lock:
            n = self._conn.execute("DELETE FROM scans").rowcount
            self._conn.commit()
        return n

    def delete(self, scan_id: str) -> bool:
        with self._lock:
            n = self._conn.execute("DELETE FROM scans WHERE id = ?", (scan_id,)).rowcount
            self._conn.commit()
        return n > 0

    def stats(self, days: int = 14) -> dict[str, Any]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT created_at, verdict, risk_level, risk_score, suffix, result_json FROM scans"
            ).fetchall()
        total = len(rows)
        by_level = Counter(r["risk_level"] for r in rows)
        by_verdict = Counter(r["verdict"] for r in rows)

        today = datetime.now(UTC).date()
        timeline = {(today - timedelta(days=i)).isoformat(): {"phishing": 0, "legitimate": 0} for i in range(days)}
        tld_counter: Counter[str] = Counter()
        tld_phish: Counter[str] = Counter()
        indicator_counter: Counter[str] = Counter()
        score_bins = [0] * 10
        for r in rows:
            day = r["created_at"][:10]
            if day in timeline:
                timeline[day][r["verdict"]] += 1
            tld = (r["suffix"] or "-").rsplit(".", 1)[-1] or "-"
            tld_counter[tld] += 1
            if r["verdict"] == "phishing":
                tld_phish[tld] += 1
                for ind in json.loads(r["result_json"]).get("indicators", []):
                    indicator_counter[ind["kind"]] += 1
            score_bins[min(int(r["risk_score"] // 10), 9)] += 1

        return {
            "total_scans": total,
            "phishing": by_verdict.get("phishing", 0),
            "legitimate": by_verdict.get("legitimate", 0),
            "detection_rate": (by_verdict.get("phishing", 0) / total) if total else 0.0,
            "average_risk_score": (sum(r["risk_score"] for r in rows) / total) if total else 0.0,
            "by_risk_level": {lvl: by_level.get(lvl, 0) for lvl in ("low", "medium", "high", "critical")},
            "timeline": [{"date": d, **v} for d, v in sorted(timeline.items())],
            "top_tlds": [
                {"tld": t, "scans": n, "phishing": tld_phish.get(t, 0)} for t, n in tld_counter.most_common(8)
            ],
            "top_indicators": [{"kind": k, "count": n} for k, n in indicator_counter.most_common(8)],
            "score_histogram": [{"range": f"{i * 10}-{i * 10 + 10}", "count": c} for i, c in enumerate(score_bins)],
        }
