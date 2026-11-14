# Model card — SentinelAI XGBoost intrusion classifier

Every number in this card is copied from `backend/models/metrics.json`,
produced by `python -m sentinelai.training.train --cross-dataset` (seed 42,
Python 3.13, scikit-learn 1.9.1, XGBoost 3.4.1, 4 CPU threads, 23.8 min).
Re-running the command regenerates them. Small differences can appear across
CPU architectures or library versions.

## Model details

| | |
|---|---|
| Model | XGBoost multi-class (`multi:softprob`), 120 boosting rounds × 8 classes, depth 8, η 0.1, subsample/colsample 0.8, early stopping on validation weighted log-loss |
| Input | 47 CICFlowMeter bidirectional flow statistics (out of 77). Either CIC-IDS2017 or CSE-CIC-IDS2018 column naming is accepted |
| Output | Calibrated probability for each of 8 classes. Severity: class base impact, lowered one level when confidence < 0.6 |
| Calibration | Temperature scaling, T = 1.053 (fitted on validation) |
| Artifact | `sentinel_xgb.ubj` (3.4 MB, XGBoost native format) + `model_meta.json`; SHA-256 verified at load; no pickle |
| Explanations | Exact TreeSHAP (SHAP `TreeExplainer`); bulk reasons via XGBoost's built-in TreeSHAP (tested to be identical) |
| Version | 1.0.0 |

## Intended use

* **Intended:** decision support for analysing exported flow records (CSV), for
  teaching and research on ML-based intrusion detection, and for demonstrating
  explainable detection.
* **Not intended:** autonomous blocking, an inline production IDS, or use on
  traffic that differs substantially from CIC-IDS2017 without retraining. See
  *Generalisation* below. That section is the most important part of this card.

## Training data

**CIC-IDS2017** (Sharafaldin et al., ICISSP 2018). Five days of emulated
enterprise traffic with scripted attacks, exported by CICFlowMeter. The archive
is verified by SHA-256. The per-label counts must equal the published counts
(2,830,743 flows).

| Step | Flows |
|---|---|
| Raw | 2,830,743 |
| Exact duplicate feature vectors removed (before splitting) | −302,594 |
| Contradictory vectors removed (697 vectors with > 1 label) | −7,018 |
| Heartbleed (11) excluded; kept as an unseen-attack probe | — |
| **Clean, used for train/val/test** | **2,521,120** |

Stratified 70/15/15 split: 1,764,784 / 378,168 / 378,168 flows.

| Class | Raw labels | Test flows |
|---|---|---|
| BENIGN | BENIGN | 314,392 |
| DoS | DoS Hulk, GoldenEye, slowloris, Slowhttptest | 29,043 |
| DDoS | DDoS (LOIC) | 19,202 |
| PortScan | PortScan | 13,538 |
| BruteForce | FTP-Patator, SSH-Patator | 1,373 |
| WebAttack | Web Attack – Brute Force / XSS / Sql Injection | 322 |
| Bot | Bot (Ares) | 293 |
| Infiltration | Infiltration | **5** |

Class weights: (n_max/n_c)^0.5, capped at 100 (Bot 32.8, Infiltration 100).

## Evaluation — primary protocol (stratified split, test set)

### Model comparison

| Model | Val macro-F1 | Test macro-F1 | Accuracy | Macro ROC-AUC (OvR) | Attack recall | FPR | Train | Inference | Size |
|---|---|---|---|---|---|---|---|---|---|
| Random Forest (100 trees) | **0.9584** | **0.9400** | 99.90% | 0.9866 | 99.80% | **0.080%** | 116 s | 2.4 µs/flow | 87.9 MB (pickle) |
| **XGBoost (deployed)** | 0.9550 | 0.9145 | 99.90% | 0.9749 | **99.94%** | 0.102% | 119 s | 5.2 µs/flow | 3.4 MB* |
| Decision Tree (depth 25) | 0.9494 | 0.9051 | 99.90% | 0.9674 | 99.91% | 0.095% | 58 s | 0.1 µs/flow | 0.3 MB |
| Logistic Regression | 0.7606 | 0.7570 | 98.67% | 0.9914 | 98.61% | 1.296% | 200 s | 0.7 µs/flow | 4 kB |
| Isolation Forest (benign-only) | — | — | — | — | 1.12%** | 1.022% | 3 s | 13.9 µs/flow | 6.8 MB |

\* 4.2 MB as trained. The deployed booster is truncated to the early-stopping
optimum.
\*\* At a 1% false-alarm budget (threshold = 99th percentile of benign
validation scores). Binary ROC-AUC is 0.9218. At the F1-optimal threshold,
which uses validation labels and is shown for context only, recall is 79.1%
with a 10.6% FPR.

Inference time is batch scoring of the test split. The API adds parsing and
storage: 3,905 flows end-to-end take ≈1.6 s.

### Deployed model, per class

| Class | Precision | Recall (95% Wilson CI) | F1 | ROC-AUC |
|---|---|---|---|---|
| BENIGN | 0.9999 | 0.9990 (0.999–0.999) | 0.9994 | 0.99998 |
| DoS | 0.9975 | 0.9992 (0.999–1.000) | 0.9984 | 0.99999 |
| DDoS | 0.9999 | 0.9998 (1.000–1.000) | 0.9998 | 1.00000 |
| PortScan | 0.9957 | 0.9995 (0.999–1.000) | 0.9976 | 0.99998 |
| BruteForce | 1.0000 | 0.9985 (0.995–1.000) | 0.9993 | 1.00000 |
| WebAttack | 0.9489 | 0.9814 (0.960–0.991) | 0.9649 | 0.99999 |
| Bot | 0.6159 | 0.9795 (0.956–0.991) | 0.7563 | 0.99991 |
| Infiltration | 0.6000 | 0.6000 (0.231–0.882) | 0.6000 | 0.7993 |

Attack-vs-benign view: 63,737 attacks caught, 39 missed, and 321 false alarms
among 314,392 benign flows (FPR 0.102%, ROC-AUC 0.99998).

### Why XGBoost is deployed although Random Forest scores higher

The selection rule was fixed before looking at the test set: highest
validation macro-F1, subject to pickle-free serialisation and exact TreeSHAP.
On validation, Random Forest leads by 0.0034 macro-F1. On test, the gap is
0.026. It comes from two classes:

* **Bot** — different points on the same precision/recall trade-off. XGBoost:
  P 0.62 / R 0.98. RF: P 0.73 / R 0.90. The Bot class weight (32.8) pushes
  XGBoost towards recall.
* **Infiltration** — 5 test flows. RF classifies 3/5 with no false positives;
  XGBoost classifies 3/5 with 2 false positives. Each flow moves F1 by about
  0.1, which makes macro-F1 dominated by noise from this class.

XGBoost is equal or better on DoS, DDoS, PortScan and BruteForce, and has
higher attack recall (99.94% vs 99.80%). It is 26× smaller and loads without
pickle (Random Forest would require loading an 88 MB pickle). For an
analyst-facing tool, the trade is a slightly higher false-alarm rate on benign
traffic misread as Bot. That is documented, not hidden. Tuning the class
weights for Bot on validation is listed as future work.

### Calibration

Top-label ECE on test: 0.00037 → 0.00027 after temperature scaling. NLL:
0.003829 → 0.003806. The raw model was already close to calibrated. 99.8% of
test flows receive confidence above 0.93, where the empirical accuracy is
99.94%. Temperature scaling does not change any prediction (asserted in the
pipeline).

## Robustness — temporal split

Within every (day, class) block, the earliest 70% of flows train and the latest
15% test. Feature selection and preprocessing are refitted.

| Model | Macro-F1 random split | Macro-F1 temporal split | Attack recall | FPR |
|---|---|---|---|---|
| Random Forest | 0.9400 | 0.7953 | 85.86% | 0.045% |
| XGBoost | 0.9145 | 0.7505 | 88.73% | 0.070% |
| Decision Tree | 0.9051 | 0.7739 | 90.31% | 0.064% |
| Logistic Regression | 0.7570 | 0.6570 | 78.68% | 0.624% |

XGBoost per-class recall, random → temporal: DoS 99.9% → 73.9%, BruteForce
99.9% → 94.8%, WebAttack 98.1% → 86.3%, **Bot 98.0% → 0.0%** (RF also 0.0%),
DDoS and PortScan stay above 99%.

**Interpretation.** Random splits interleave consecutive flows of the same
attack run between train and test. When the model must recognise *later*
traffic of an attack, it degrades sharply for attacks whose flow statistics
drift during the run (Bot C2 beacons, DoS tools). The random-split numbers
above are an upper bound and should not be quoted as expected field
performance.

## Generalisation — CSE-CIC-IDS2018

Seven 2018 day files whose attacks overlap the taxonomy. At most 40,000 flows
per (day, label) are sampled, de-duplicated (442,385 flows), and split
chronologically. PortScan is absent from 2018. Macro-F1 is computed over the
classes present.

| Trained on | 2017 test macro-F1 | 2017 attack recall | 2018 test macro-F1 | 2018 attack recall | 2018 FPR |
|---|---|---|---|---|---|
| CIC-IDS2017 (deployed) | 0.9145 | 99.9% | **0.2223** | 26.0% | 0.06% |
| CSE-CIC-IDS2018 only | 0.1690 | 18.1% | 0.7655 | 67.4% | 6.35% |
| 2017 + 2018 jointly | 0.8374 | 99.9% | 0.7937 | 88.7% | **30.4%** |
| 2017 without host TCP-stack features | 0.8659 | 99.4% | 0.2204 | 27.1% | 1.64% |

Per-class recall of the deployed model on 2018 traffic: DoS 60.9%, and **0%**
for DDoS, BruteForce, WebAttack, Bot and Infiltration.

**Findings.**

1. **A CIC-IDS2017 model does not transfer.** The same attack *types*,
   generated with similar tools on another network and with a newer
   CICFlowMeter, are mostly classified as benign. The reverse direction fails
   in the same way.
2. **Adding 2018 data is not a free win.** The joint model covers both test
   sets, but it falsely flags 30.4% of 2018 benign flows, and its 2017
   macro-F1 drops from 0.914 to 0.837. Combining datasets mainly widens
   coverage; it does not produce a model that generalises to a *third*
   network. That claim would need a third dataset to test.
3. **Removing the TCP-stack features does not help transfer.** The initial
   window sizes and minimum segment size dominate the SHAP ranking. These
   features describe the hosts rather than the attack behaviour, so they were
   suspected shortcuts. The ablation shows they are not the main cause of the
   transfer failure.
4. **The runtime shift indicator flags 2018 traffic as "high" shift.** Up to
   23.6% of a feature's values (`RST Flag Count`) fall outside the 2017
   training range. The dashboard shows this warning on such uploads.

## Unseen attack probe

The 11 Heartbleed flows (never in training) were all classified as BENIGN by
XGBoost (0/11 flagged). Isolation Forest flagged **11/11**. A supervised
classifier only recognises what it was trained on. Pairing it with an anomaly
detector (despite that detector's weak recall at low FPR on known attacks) is a
sensible next step.

## Explainability findings (global SHAP, 1,755 class-balanced test flows)

Top features by mean |SHAP| on the predicted class:

1. `Init_Win_bytes_backward`
2. `Init_Win_bytes_forward`
3. `Bwd Header Length`
4. `Flow Bytes/s`
5. `act_data_pkt_fwd`
6. `min_seg_size_forward`
7. `Total Length of Fwd Packets`
8. `Flow IAT Max`

Per class:

| Class | Leading features |
|---|---|
| PortScan | `Total Length of Fwd Packets`, `Packet Length Mean`, `Flow Bytes/s` — tiny payload-less probes |
| DDoS | `act_data_pkt_fwd`, `Total Length of Bwd Packets`, `Fwd Packet Length Mean` |
| BruteForce | `Bwd Header Length`, `min_seg_size_forward`, `Fwd PSH Flags` |
| Infiltration | `Packet Length Mean`, `Total Length of Fwd Packets` — large forward transfers |

The prominence of TCP-window features is a known property of CIC-IDS2017.
The cross-dataset ablation above tests whether they are shortcuts.

## Limitations

* **Lab data.** CIC-IDS2017 is emulated traffic with one attacker network and
  scripted attacks. Published analyses (e.g. Engelen et al., 2021, and Lanvin et
  al., 2022) document labelling and flow-construction errors in the dataset.
  This project removes duplicates and contradictory labels but does not
  relabel flows.
* **Tiny classes.** Infiltration has 5 test flows (recall 95% CI 23–88%). Its
  metrics carry almost no statistical weight.
* **Optimistic headline metrics.** See the temporal and cross-dataset results.
  Expect substantially lower performance on other networks without retraining.
* **Bot false alarms.** Bot precision is 0.62 under the chosen class weights.
* **Flow-level only.** No payload inspection, and no correlation across flows
  or hosts.

## Ethical considerations

* Uploaded traffic can contain personal data. The system drops IP addresses,
  ports, flow identifiers and timestamps on ingestion and stores only
  statistical flow features. History is bounded and deletable.
* A false negative is a missed attack and a false positive costs analyst time.
  The UI therefore shows calibrated confidence, severity, an explanation, a
  distribution-shift warning, and a standing reminder that predictions are
  decision support.

## Reproducing

```bash
cd backend
pip install -r requirements-dev.txt
python -m sentinelai.training.train --cross-dataset   # downloads ~2.6 GB, ~25 min on 4 cores
python scripts/make_samples.py
```

The dataset archives are fetched from pinned URLs and verified by SHA-256
before use.
