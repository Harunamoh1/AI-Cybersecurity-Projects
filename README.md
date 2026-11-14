# AI-Cybersecurity-Projects

AI-powered cybersecurity research and software engineering projects.

| Project | Description | Stack |
|---|---|---|
| [**PhishLens: AI Phishing Detection System**](phishing-detection/) | Explainable ML phishing-URL detection. Trained and evaluated on about 260k URLs with a domain-grouped split (test ROC-AUC 0.921, F1 0.837), served by a FastAPI API and shown in a Next.js security console. | Python · scikit-learn · FastAPI · Next.js · TypeScript · Tailwind · Recharts · Docker · Pytest |
| [**SentinelAI: Explainable Network Intrusion Detection**](network-intrusion-detection/) | 8-class NIDS on CIC-IDS2017 flow features with exact SHAP explanations. Test macro-F1 0.915 / attack recall 99.94% (stratified), with honest temporal-split (0.751) and CSE-CIC-IDS2018 cross-dataset (0.222) results, served by a FastAPI API and a Next.js analyst console. | Python · XGBoost · scikit-learn · SHAP · FastAPI · Next.js · TypeScript · Tailwind · Recharts · Docker · Pytest |

Each project lives in its own directory with its own README, tests and Docker setup.
