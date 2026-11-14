"""SQLite store for analyses (uploaded files / API batches) and their flows.

Privacy by design: only the model's input features are persisted. Host
identifiers (IP addresses, ports, flow ids, timestamps) are dropped on
ingestion and never reach this store. All queries are parameterised. The
database file is created with mode 0600 and the number of retained analyses
is bounded (oldest are pruned).
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id              TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL,
    filename        TEXT NOT NULL,
    source          TEXT NOT NULL,
    total_flows     INTEGER NOT NULL,
    threats         INTEGER NOT NULL,
    rejected_rows   INTEGER NOT NULL,
    duration_ms     REAL NOT NULL,
    model_version   TEXT NOT NULL,
    summary_json    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_analyses_created ON analyses(created_at DESC);

CREATE TABLE IF NOT EXISTS flows (
    analysis_id     TEXT NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
    idx             INTEGER NOT NULL,
    created_at      TEXT NOT NULL,
    predicted       TEXT NOT NULL,
    confidence      REAL NOT NULL,
    attack_prob     REAL NOT NULL,
    severity        TEXT NOT NULL,
    true_label      TEXT,
    imputed         INTEGER NOT NULL,
    reasons_json    TEXT,
    features        BLOB NOT NULL,
    PRIMARY KEY (analysis_id, idx)
);
CREATE INDEX IF NOT EXISTS idx_flows_threat ON flows(predicted, severity);
CREATE INDEX IF NOT EXISTS idx_flows_created ON flows(created_at DESC);
"""

SEVERITY_RANK_SQL = "CASE severity WHEN 'critical' THEN 4 WHEN 'high' THEN 3 WHEN 'medium' THEN 2 WHEN 'low' THEN 1 ELSE 0 END"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: str, retention: int = 200):
        self.path = path
        self.retention = retention
        self._lock = threading.Lock()
        if path != ":memory:" and not os.path.exists(path):
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            fd = os.open(path, os.O_CREAT | os.O_WRONLY, 0o600)
            os.close(fd)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys=ON")
        if path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------ writes
    def add_analysis(self, *, filename: str, source: str, summary: dict[str, Any], flows: list[dict[str, Any]],
                     features: np.ndarray, rejected_rows: int, duration_ms: float, model_version: str) -> dict[str, Any]:
        analysis_id = uuid.uuid4().hex
        created = _now()
        threats = sum(1 for f in flows if f["predicted_class"] != "BENIGN")
        feats = np.ascontiguousarray(features, dtype=np.float32)
        rows = [
            (analysis_id, f["index"], created, f["predicted_class"], f["confidence"], f["attack_probability"],
             f["severity"], f.get("true_label"), f.get("imputed_features", 0),
             json.dumps(f["reasons"]) if f.get("reasons") else None, feats[i].tobytes())
            for i, f in enumerate(flows)
        ]
        with self._lock:
            self._conn.execute(
                "INSERT INTO analyses (id, created_at, filename, source, total_flows, threats, rejected_rows,"
                " duration_ms, model_version, summary_json) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (analysis_id, created, filename, source, len(flows), threats, rejected_rows, duration_ms,
                 model_version, json.dumps(summary)),
            )
            self._conn.executemany("INSERT INTO flows VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
            self._prune()
            self._conn.commit()
        return self.get_analysis(analysis_id)  # type: ignore[return-value]

    def _prune(self) -> None:
        if self.retention <= 0:
            return
        self._conn.execute(
            "DELETE FROM analyses WHERE id IN (SELECT id FROM analyses ORDER BY created_at DESC, rowid DESC"
            " LIMIT -1 OFFSET ?)", (self.retention,))

    def delete_analysis(self, analysis_id: str) -> bool:
        with self._lock:
            n = self._conn.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,)).rowcount
            self._conn.commit()
        return n > 0

    def clear(self) -> int:
        with self._lock:
            n = self._conn.execute("DELETE FROM analyses").rowcount
            self._conn.commit()
            self._conn.execute("VACUUM")
        return n

    # ------------------------------------------------------------------- reads
    @staticmethod
    def _analysis(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        d["summary"] = json.loads(d.pop("summary_json"))
        return d

    def get_analysis(self, analysis_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
        return self._analysis(row) if row else None

    def list_analyses(self, limit: int, offset: int) -> tuple[int, list[dict[str, Any]]]:
        with self._lock:
            total = self._conn.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
            rows = self._conn.execute(
                "SELECT * FROM analyses ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?", (limit, offset)
            ).fetchall()
        return total, [self._analysis(r) for r in rows]

    @staticmethod
    def _flow(row: sqlite3.Row, with_features: bool = False) -> dict[str, Any]:
        d = {k: row[k] for k in row.keys() if k not in ("features", "reasons_json")}
        d["reasons"] = json.loads(row["reasons_json"]) if row["reasons_json"] else []
        if with_features:
            d["features"] = np.frombuffer(row["features"], dtype=np.float32)
        return d

    def list_flows(self, *, analysis_id: str | None = None, threats_only: bool = False, cls: str | None = None,
                   severity: str | None = None, min_confidence: float | None = None, limit: int = 50,
                   offset: int = 0, order: str = "index") -> tuple[int, list[dict[str, Any]]]:
        where, params = [], []
        if analysis_id:
            where.append("f.analysis_id = ?")
            params.append(analysis_id)
        if threats_only:
            where.append("f.predicted != 'BENIGN'")
        if cls:
            where.append("f.predicted = ?")
            params.append(cls)
        if severity:
            where.append("f.severity = ?")
            params.append(severity)
        if min_confidence is not None:
            where.append("f.confidence >= ?")
            params.append(min_confidence)
        clause = (" WHERE " + " AND ".join(where)) if where else ""
        order_sql = {
            "index": "f.idx ASC",
            "severity": f"{SEVERITY_RANK_SQL.replace('severity', 'f.severity')} DESC, f.confidence DESC, f.idx ASC",
            "recent": f"f.created_at DESC, {SEVERITY_RANK_SQL.replace('severity', 'f.severity')} DESC, f.idx ASC",
        }[order]
        with self._lock:
            total = self._conn.execute(f"SELECT COUNT(*) FROM flows f{clause}", params).fetchone()[0]  # noqa: S608
            rows = self._conn.execute(
                "SELECT f.analysis_id, f.idx, f.created_at, f.predicted, f.confidence, f.attack_prob, f.severity,"  # noqa: S608
                f" f.true_label, f.imputed, f.reasons_json, a.filename FROM flows f JOIN analyses a ON a.id = f.analysis_id"
                f"{clause} ORDER BY {order_sql} LIMIT ? OFFSET ?",
                [*params, limit, offset],
            ).fetchall()
        return total, [self._flow(r) for r in rows]

    def get_flow(self, analysis_id: str, idx: int) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT f.*, a.filename FROM flows f JOIN analyses a ON a.id = f.analysis_id"
                " WHERE f.analysis_id = ? AND f.idx = ?", (analysis_id, idx)).fetchone()
        return self._flow(row, with_features=True) if row else None

    def stats(self, days: int = 14) -> dict[str, Any]:
        since = (datetime.now(UTC).date() - timedelta(days=days - 1)).isoformat()
        with self._lock:
            c = self._conn
            totals = c.execute("SELECT COUNT(*) AS analyses, COALESCE(SUM(total_flows),0) AS flows,"
                               " COALESCE(SUM(threats),0) AS threats FROM analyses").fetchone()
            by_class = c.execute("SELECT predicted, COUNT(*) AS n, AVG(confidence) AS conf FROM flows GROUP BY predicted").fetchall()
            by_sev = c.execute("SELECT severity, COUNT(*) AS n FROM flows WHERE predicted != 'BENIGN' GROUP BY severity").fetchall()
            trend = c.execute(
                "SELECT substr(created_at, 1, 10) AS day, predicted, COUNT(*) AS n FROM flows"
                " WHERE created_at >= ? GROUP BY day, predicted", (since,)).fetchall()
            conf_bins = c.execute(
                "SELECT MIN(CAST(confidence * 10 AS INTEGER), 9) AS b, COUNT(*) AS n FROM flows"
                " WHERE predicted != 'BENIGN' GROUP BY b").fetchall()
            gt = c.execute("SELECT COUNT(*) AS n, SUM(CASE WHEN true_label = predicted THEN 1 ELSE 0 END) AS ok"
                           " FROM flows WHERE true_label IS NOT NULL").fetchone()
        today = datetime.now(UTC).date()
        timeline: dict[str, dict[str, int]] = {
            (today - timedelta(days=i)).isoformat(): {} for i in range(days - 1, -1, -1)}
        for r in trend:
            if r["day"] in timeline:
                timeline[r["day"]][r["predicted"]] = r["n"]
        return {
            "analyses": totals["analyses"],
            "total_flows": totals["flows"],
            "threats": totals["threats"],
            "threat_rate": (totals["threats"] / totals["flows"]) if totals["flows"] else 0.0,
            "by_class": {r["predicted"]: {"count": r["n"], "mean_confidence": r["conf"]} for r in by_class},
            "by_severity": {r["severity"]: r["n"] for r in by_sev},
            "timeline": [{"date": d, "total": sum(v.values()), "threats": sum(n for k, n in v.items() if k != "BENIGN"),
                          "by_class": v} for d, v in timeline.items()],
            "threat_confidence_histogram": [{"bin": f"{b / 10:.1f}-{(b + 1) / 10:.1f}",
                                             "count": next((r["n"] for r in conf_bins if r["b"] == b), 0)} for b in range(10)],
            "labelled_flows": gt["n"] or 0,
            "labelled_agreement": (gt["ok"] / gt["n"]) if gt["n"] else None,
        }
