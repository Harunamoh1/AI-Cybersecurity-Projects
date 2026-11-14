# Model Card: PhishLens URL Classifier v1.0.0

Every figure on this page comes from `backend/models/metrics.json`, which `python -m phishlens.ml.train` produced on 2026-11-14 (scikit-learn 1.9.1, Python 3.13).

## Model details

| | |
|---|---|
| Type | `HistGradientBoostingClassifier` (400 iterations, learning rate 0.08, 63 max leaf nodes, min 40 samples per leaf, L2 = 1.0) in a scikit-learn `Pipeline` with a `ColumnTransformer` |
| Inputs | One URL string. It is canonicalised and turned into 35 numeric features plus the TLD (target-encoded with 5-fold cross-fitting). |
| Output | Phishing probability `p`. Verdict = `p ≥ 0.4172`, the threshold that maximised F1 on validation. |
| Explanations | Occlusion in log-odds space against the median feature values of legitimate training URLs |
| Artifact | `phishlens_model.joblib` (1.28 MB), SHA-256 `dff2e7b2…0f06`, checked before loading |

## Intended use

- **Use it for:** triage and decision support in a phishing-analysis workflow, and as a teaching or portfolio example of leakage-aware URL classification.
- **Do not use it for:** automatic blocking without human review or other signals, judging page content, attachments or e-mail headers, or treating a low score as proof that a URL is safe.

## Training data

| Source | Clean rows used | Composition |
|---|---|---|
| PhiUSIIL (UCI #967, CC BY 4.0) | ~232k | Phishing URLs plus legitimate *home pages* |
| Hannousse & Yahiouche (Mendeley, CC BY 4.0) | ~11k | Balanced; legitimate URLs include paths |
| Legitimate deep URLs (Tranco `Y897G` × Common Crawl `CC-MAIN-2026-39`) | 16,165 | Legitimate pages with paths from 1,790 popular domains |

Cleaning removed:

- 1 unparseable row
- 86 rows with contradicting labels
- 3,904 canonical duplicates

That leaves 259,399 rows: 102,889 phishing and 156,510 legitimate.

**Splits.** `StratifiedGroupKFold` grouped by registrable domain (Public Suffix List, including private suffixes) and stratified on label + source:

| Split | Rows | Domains |
|---|---|---|
| Train | 166,014 | 131,541 |
| Validation | 41,504 | 32,890 |
| Test | 51,881 | 41,110 |

No domain appears in more than one split. The training run asserts this.

**Weighting.** Each (source, label) cell gets equal total sample weight.

## Known dataset artefacts and how they were handled

1. **Every PhiUSIIL legitimate URL is `https://www.<domain>`, with no path and no trailing slash, and the dataset is lower-cased.** Without intervention, scheme, `www.`, trailing slashes, letter case and "has a path" would each separate the classes on their own, without any security meaning. PhishLens canonicalises URLs: it lower-cases them and drops the scheme, `www.`, the fragment and a bare `/`. It also adds a corpus of legitimate deep URLs.
2. **PhiUSIIL's pre-computed columns** include HTML-derived features and a similarity index computed against the legitimate set. They are not used.
3. **Popular-domain bias in the added corpus.** Reputation features such as Tranco rank are deliberately left out, because the corpus was sampled from Tranco and the feature would leak the label.

## Evaluation (test split, threshold 0.4172)

| Metric | Value |
|---|---|
| Accuracy | 0.8798 |
| Precision | 0.9037 |
| Recall | 0.7800 |
| F1 | 0.8373 |
| ROC-AUC | 0.9211 |
| PR-AUC | 0.9203 |
| Brier score | 0.0922 |
| False-positive rate | 5.46% |
| False-negative rate | 22.00% |

Confusion matrix: TN 29,593 · FP 1,710 · FN 4,527 · TP 16,051.

### By slice

| Slice | FPR | Recall | ROC-AUC |
|---|---|---|---|
| PhiUSIIL | 3.8% | 77.5% | 0.919 |
| Hannousse | 20.9% | 87.1% | 0.916 |
| Legitimate deep URLs | 14.0% | n/a | n/a |

### Experiments

| Experiment | F1 | ROC-AUC | Deep-URL FPR |
|---|---|---|---|
| Deployed | 0.837 | 0.921 | 14.0% |
| Without source weights | 0.841 | 0.925 | 19.3% |
| Without the legitimate deep-URL corpus | 0.824 | 0.914 | 45.7% |
| Train PhiUSIIL only → test Hannousse | 0.725 | 0.682 | n/a |
| Train Hannousse only → test PhiUSIIL | 0.772 | 0.872 | n/a |

The cross-dataset rows are the most important result on this page: performance on data from a different source is much lower than in-distribution scores suggest.

## Ethical considerations and risks

- **False positives** can block legitimate businesses. The risk is highest for:
  - small sites on uncommon TLDs
  - free-hosting platforms
  - URLs with long random tokens
  - login pages of real brands, because the model has no reputation signal

  Examples on the live model: `accounts.google.com/signin/v2/identifier` scores 0.60 and `paypal.com/signin` scores 0.44, both above the threshold.
- **False negatives** (22%) give false reassurance. The UI says that a verdict is decision support.
- **Language and region bias:** the brand and keyword lists are mostly English and Western, so local brands are not covered.
- **Drift:** retrain on fresh feeds, and re-run the cross-dataset experiments when you do.

## Reproducing

```bash
cd backend
```

```bash
pip install -r requirements-dev.txt
```

```bash
python -m phishlens.ml.train --data-dir data --out-dir models
```

The datasets are fetched and SHA-256 verified, and every random step (split, models, target encoder, permutation importance) is seeded with 42, so the split is deterministic. The trained model should be identical on the same library versions; I have not verified byte-identical artifacts across machines. Different library versions may give small numerical differences.
