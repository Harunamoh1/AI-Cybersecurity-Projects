# SentinelAI — Architecture

This document records how the system is put together and *why*. Results live in
[`MODEL_CARD.md`](MODEL_CARD.md); setup lives in the [README](../README.md).

## 1. System overview

```
                ┌───────────────────────── offline (reproducible) ──────────────────────────┐
 CIC-IDS2017 ──►│ sources ─► loading ─► cleaning ─► splitting ─► features ─► training ─► evaluation │──► models/
 (pinned,       │ (sha256)   (verify    (dedupe,     (stratified  (select,   (5 models)   (metrics,  │    sentinel_xgb.ubj
  sha256)       │            counts)    conflicts)   + temporal)  impute)                 SHAP,      │    model_meta.json
 CSE-CIC-       │                                                                         cross-     │    SHA256SUMS
 IDS2018 ──────►│ ──────────────────────────── crossdataset experiments ───────────────── dataset)  │    metrics.json
                └──────────────────────────────────────────────────────────────────────────────────┘
                                                                                             │
                ┌──────────────────────────────── online ─────────────────────────────────────▼─────┐
  Browser ─────►│ Next.js (UI + /api proxy) ──► FastAPI ──► service ──► Detector ──► XGBoost + SHAP │
  (CSV upload)  │   adds API key server-side      │  upload.py (defensive CSV parsing)              │
                │                                 └► Store (SQLite, features only, no IPs)          │
                └────────────────────────────────────────────────────────────────────────────────────┘
```

| Layer | Module | Responsibility |
|---|---|---|
| Schema | `sentinelai/schema.py` | 77 canonical flow features with CIC-IDS2017 *and* CSE-CIC-IDS2018 aliases; the 8-class taxonomy; raw-label mapping; severities |
| Data | `sentinelai/data/sources.py` | Pinned download URLs + SHA-256, HTTPS-only fetch, zip-slip-safe extraction |
| | `sentinelai/data/loading.py` | CSV reading, published-count verification, timestamp parsing, de-duplication and conflict removal |
| | `sentinelai/data/splitting.py` | Stratified and chronological (temporal) splits |
| Features | `sentinelai/features.py` | Column canonicalisation, sanitisation, train-only feature selection, median imputation — **shared by training and inference** |
| Training | `sentinelai/training/models.py` | Model factories, class weights |
| | `sentinelai/training/train.py` | Orchestrates protocols, calibration, SHAP, artifact writing |
| | `sentinelai/training/evaluate.py` | All reported metrics (incl. Wilson CIs, ECE, ROC) |
| | `sentinelai/training/calibration.py` | Temperature scaling |
| | `sentinelai/training/crossdataset.py` | CSE-CIC-IDS2018 generalisation experiments + ablation |
| Inference | `sentinelai/inference/detector.py` | Loads the artifact (checksum-verified, pickle-free), prepares input, predicts, explains, flags distribution shift |
| API | `sentinelai/api/*` | FastAPI app, config, upload hardening, SQLite store, rate limiting |
| Frontend | `frontend/src/*` | Next.js App Router dashboard; same-origin proxy to the API |

## 2. Data decisions

**Primary dataset.** CIC-IDS2017 (`GeneratedLabelledFlows.zip`, i.e. the
CICFlowMeter export *with* timestamps). The official UNB download is behind a
registration form, so the archive is fetched from a public mirror pinned to an
immutable commit. Authenticity is checked twice: the archive SHA-256, and the
per-label row counts, which must equal the published counts exactly
(2,830,743 flows). The pipeline refuses to continue otherwise.

**Taxonomy.** Fifteen raw labels map to eight classes: BENIGN, DoS, DDoS,
PortScan, BruteForce, WebAttack, Bot, Infiltration. All eight are supported by
the data. Heartbleed (11 flows) cannot be learned or evaluated meaningfully, so
it is excluded from training and kept as an *unseen-attack probe*.

**Cleaning (before any split).**

| Problem found during exploration | Treatment |
|---|---|
| 10.9% of rows are exact duplicate feature vectors | removed (keep one), *before* splitting — otherwise identical flows land in train and test |
| 697 feature vectors carry two different labels (7,018 rows) | removed entirely — irreducible label noise |
| `±inf` in `Flow Bytes/s`, `Flow Packets/s` (zero-duration flows) | → NaN → training median |
| Negative durations / IATs (capture clock artefacts) | → NaN → training median |
| `Init_Win_bytes_* = -1` | kept: documented "not observed" sentinel |
| Duplicated `Fwd Header Length` column | dropped at canonicalisation |
| Mis-encoded label text (`Web Attack � XSS`) | normalised |

**Leakage controls.** Identifiers (Flow ID, IPs, ports, timestamp, protocol)
are never features. `Destination Port` is excluded by policy: it lets a model
shortcut on lab-specific port usage (all FTP-Patator traffic is on port 21).
Feature selection, imputation medians and class weights are fitted on the
training split only. The test split is used exclusively for reporting; model
selection, XGBoost early stopping, calibration and the Isolation Forest
threshold use the validation split.

**Feature selection** (unsupervised, train split): drop constant columns, exact
duplicate columns, then near-duplicates with |Spearman ρ| ≥ 0.99. This removes
redundancy without looking at labels, and keeps SHAP from splitting credit
between copies of the same signal. 47 of 77 features remain; every drop is
logged with its reason in `metrics.json`.

## 3. Evaluation protocols

1. **Primary — stratified random 70/15/15** after de-duplication. Comparable
   with the literature, made honest by de-duplicating first.
2. **Robustness — temporal split.** Within every (capture day, class) block,
   the earliest 70% of flows train, the next 15% validate, the latest 15% test.
   Random splits interleave consecutive, near-identical flows of the same attack
   run across train and test; the temporal split removes that optimism.
3. **Generalisation — CSE-CIC-IDS2018.** (A) the deployed 2017 model on 2018
   traffic; (B) a 2018-only model, in-domain and reverse (→ 2017); (C) a joint
   model; (D) an ablation without host TCP-stack features.

Imbalance is handled with class weights `(n_max / n_c)^0.5`, capped at 100 —
a deliberate compromise between minority recall and benign false alarms. Small
classes are reported with Wilson 95% intervals rather than bare point estimates.

## 4. Models and the deployment choice

Five models are trained and compared under identical data: Random Forest,
XGBoost, Decision Tree, Logistic Regression (signed-log + standardisation) and
an Isolation Forest trained on benign flows only (anomaly-detection baseline).

The deployed model must satisfy three constraints in addition to accuracy:

* **Pickle-free persistence.** Loading a pickle executes arbitrary code. XGBoost
  boosters serialise to a native binary-JSON format (`.ubj`) and the remaining
  state (feature list, medians, temperature, reference ranges) is plain JSON.
  Nothing on the inference path calls `pickle`/`joblib`.
* **Exact, fast explanations.** TreeSHAP is exact for tree ensembles; XGBoost
  also ships a C++ TreeSHAP used for bulk "top reasons", which the tests verify
  equals the SHAP library's output.
* **Size and latency** suitable for a container (MB, not hundreds of MB).

XGBoost meets these; the Random Forest achieves slightly higher macro-F1 in
the primary protocol (see the model card for the exact numbers and the
discussion), but serialises to hundreds of MB via pickle only.

**Calibration.** A single temperature, fitted on validation by minimising NLL,
rescales the logits. It never changes the predicted class; SHAP values (logit
space) remain exactly consistent — calibrated contributions are the raw ones
divided by T.

## 5. Explainability

* **Local** (`Detector.explain`): SHAP `TreeExplainer` contributions to the
  predicted class's (temperature-scaled) logit, plus *attack evidence* —
  contributions against the BENIGN logit — which answers "why malicious"
  independent of the attack type. Each item carries the flow's value and the
  benign training median for context. Base value + Σ contributions reproduces
  the logit (tested).
* **Bulk reasons**: for every flagged flow in an upload, the top-3 positive
  contributors (XGBoost native TreeSHAP), stored with the detection.
* **Global** (computed at training time on a class-balanced test sample):
  mean |SHAP| importance, class × feature importance matrix, beeswarm of attack
  evidence, and per-class median profiles versus benign.

## 6. Runtime security

| Threat | Control |
|---|---|
| Malicious model file | SHA-256 of booster and metadata verified before load; no pickle anywhere on the inference path |
| Oversized upload / zip bombs | `Content-Length` rejected before multipart parsing; streamed size cap; row and column caps |
| Non-CSV / binary content | `.csv` extension, content-type allow-list, NUL-byte check, text decoding required |
| CSV/formula injection | Every cell is coerced to a number; nothing is evaluated; results are never exported as CSV |
| Path traversal | Uploaded filenames are display labels only (sanitised); sample files are served from a startup whitelist, never by joining user input into a path; the Next.js proxy rejects `..` segments and only forwards an allow-list of API prefixes |
| Sensitive history | IPs, ports, flow ids and timestamps are dropped on ingestion; only model features are stored; DB file mode 0600; bounded retention; delete endpoints; optional API key (`hmac.compare_digest`) injected server-side by Next.js |
| Abuse | Per-client rate limit on compute-heavy routes; batch and explanation caps |
| SQL injection | Parameterised queries only |
| Container | Non-root user, read-only root filesystem, `cap_drop: ALL`, `no-new-privileges`, memory/pid limits, backend not published to the host |
| Browser | Same-origin proxy (no permissive CORS), `nosniff`, `X-Frame-Options: DENY`, `no-referrer`, `no-store` |

## 7. Storage model

SQLite (`analyses`, `flows`). Each flow row stores the prediction, calibrated
confidence, severity, optional ground-truth label, top reasons and the 47
preprocessed feature values as a float32 blob — enough to recompute a SHAP
explanation on demand without keeping the original file. Deleting an analysis
cascades to its flows.

## 8. Severity model

Severity = the attack class's base impact (Infiltration, Bot, DDoS: critical;
DoS, BruteForce, WebAttack: high; PortScan: medium), lowered one level when the
calibrated confidence is below 0.6. It is a triage aid, not a risk score; the
mapping lives in `schema.CLASS_INFO` and `detector.severity_for`.

## 9. Known limitations of the design

* Flow features must come from CICFlowMeter (or a compatible exporter); the
  system does not capture packets itself.
* The distribution-shift indicator is a simple range check against the training
  [p0.1, p99.9] interval; it flags obvious shift (it fires on CSE-CIC-IDS2018)
  but is not a calibrated out-of-distribution detector.
* SQLite and the in-process rate limiter assume a single API instance.
