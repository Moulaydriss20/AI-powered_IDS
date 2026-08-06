# Hybrid AI-Powered Intrusion Detection System

> A weighted-fusion hybrid IDS combining supervised and unsupervised machine learning — both algorithms implemented from scratch in NumPy.

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-only-orange.svg)](https://numpy.org/)
[![License](https://img.shields.io/badge/License-Noncommercial-red.svg)](#license)
[![Dataset](https://img.shields.io/badge/Dataset-CICIDS2017-lightgrey.svg)](https://www.unb.ca/cic/datasets/ids-2017.html)

---

## Table of Contents

- [Overview](#overview)
- [Results](#results)
- [Architecture](#architecture)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Implementation Details](#implementation-details)
- [Known Limitations](#known-limitations)
- [Roadmap](#roadmap)
- [Citation & Attribution](#citation--attribution)
- [License](#license)
- [Author](#author)

---

## Overview

Supervised intrusion detection classifiers achieve high accuracy on known attacks but are structurally blind to attack families outside their training data. Unsupervised anomaly detectors catch novel deviations but produce far more false alarms.

This project fuses both into a single scalar attack score and measures what the combination actually delivers — evaluated not just on the training distribution, but on **three attack families the models have never seen**.

**Both ML algorithms are implemented entirely from scratch in NumPy.** No scikit-learn, no TensorFlow. That includes recursive tree construction, Gini impurity splitting, bootstrap aggregation, random feature subspaces, isolation-based anomaly scoring with path-length correction, and sample-weighted feature importance.

---

## Results

### In-distribution (CICIDS2017 Wednesday)

| Model | Precision | Recall | Macro F1 | FAR |
|-------|-----------|--------|----------|-----|
| Random Forest (supervised) | 0.978 | 1.000 | 0.98 | — |
| Isolation Forest (standalone, best F1) | 0.649 | 0.869 | 0.74 | — |
| **Weighted-fusion hybrid** | **0.979** | **1.000** | **0.99** | **1.0%** |

On known attacks the fusion matches the Random Forest — expected, since the supervised model already solves this.

### Zero-day (CICIDS2017 Friday — unseen attack families)

| Attack Family | Model | Recall | FAR | Precision |
|---------------|-------|--------|-----|-----------|
| **DDoS** | RF alone | 19.2% | 0.84% | 96.8% |
| | IF alone (τ=0.447) | 63.3% | 37.6% | 69.4% |
| | **Fusion hybrid** | **75.1%** | **11.3%** | **89.9%** |
| **Port Scan** | RF alone | 0.16% | 0.13% | 48.1% |
| | IF alone (τ=0.447) | 1.20% | 17.3% | 4.8% |
| | **Fusion hybrid** | **2.97%** | **1.65%** | **57.0%** |
| **Botnet** | RF alone | 0.00% | 0.24% | 0.00% |
| | IF alone (τ=0.447) | 5.08% | 17.1% | 0.32% |
| | **Fusion hybrid** | **1.13%** | **1.80%** | **0.67%** |

On DDoS attacks the model has never seen, the fusion detects **96,101 attack flows** the supervised classifier alone would miss — a **3.9× recall improvement**.

Coverage is not uniform. The fusion works well on volumetric attacks that stand out geometrically in the flow-feature space (DDoS), and poorly on stealthy traffic that deliberately mimics benign behaviour (Botnet). This is measured and reported, not hidden.

---

## Architecture

```
                        ┌─────────────────┐
                        │  Network Flow x │
                        │  (77 features)  │
                        └────────┬────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
      ┌─────────────────────┐        ┌──────────────────────┐
      │   Random Forest     │        │   Isolation Forest   │
      │   (supervised)      │        │   (unsupervised)     │
      │  predict_proba(x)   │        │      score(x)        │
      │         ▼           │        │          ▼           │
      │  p_attack ∈ [0,1]   │        │  s_anomaly ∈ [0,1]   │
      └──────────┬──────────┘        └──────────┬───────────┘
                 └──────────────┬───────────────┘
                                ▼
              ┌──────────────────────────────────────┐
              │  c(x) = w_RF · p_attack + w_IF · s   │
              │        w_RF = 0.9,  w_IF = 0.1       │
              └──────────────────┬───────────────────┘
                                 ▼
                    ┌────────────────────────┐
                    │   c(x) ≥ τ  →  ATTACK  │
                    │      τ = 0.0558        │
                    └────────────────────────┘
```

**Parameters are not hand-tuned.** Weights come from a grid search across 9 configurations on the Wednesday test partition. The threshold is the 99th percentile of combined scores on Wednesday *benign* traffic, bounding the false-alarm rate to ~1% by construction — attack-agnostic by design.

---

## Installation

**Requirements:** Python 3.13+, NumPy, pandas (data loading only), joblib, matplotlib

```bash
git clone https://github.com/<your-username>/AI-powered_IDS.git
cd AI-powered_IDS

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install numpy pandas joblib matplotlib
```

---

## Dataset Setup

Download [CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) from the Canadian Institute for Cybersecurity and place the CSVs in `datasets/raw/`:

```
datasets/raw/
├── Wednesday-workingHours.pcap_ISCX.csv                    # Training data
├── Friday-WorkingHours-Morning.pcap_ISCX.csv               # Botnet
├── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv    # Port Scan
└── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv        # DDoS
```

The dataset is not bundled with this repository.

---

## Usage

```bash
# 1. Preprocess — clean, encode, 80/20 stratified split, fit scaler
python train/preprocess.py

# 2. Train Random Forest (~25–30 min on 6 cores)
python train/train_rf.py

# 3. Train Isolation Forest (seed=42 for reproducibility)
python train/train2.py

# 4. Grid search fusion weights + compute dynamic threshold
python train/train5.py

# 5. Evaluate on unseen Friday attack families
python train/train_fr.py

# 6. Generate the standalone IF precision-recall analysis
python evaluate/analyses.py
```

---

## Project Structure

```
AI-powered_IDS/
│
├── src/
│   ├── models/
│   │   ├── decision_tree.py            # Gini impurity, recursive splitting
│   │   ├── random_forest.py            # Bagging, feature subspaces, predict_proba
│   │   ├── isolation_tree.py           # Random partitioning
│   │   ├── isolation_forest.py         # Path-length anomaly scoring
│   │   └── weighted_hybrid_IDS.py      # Weighted-fusion classifier
│   │
│   ├── preprocessing/
│   │   ├── cleaner.py                  # Deduplication, NaN/Inf handling
│   │   ├── encoder.py                  # Label encoding
│   │   └── spliter.py                  # Stratified train-test split
│   │
│   └── feature/
│       └── scaler.py                   # StandardScaler
│
├── train/
│   ├── preprocess.py
│   ├── train_rf.py
│   ├── train2.py                       # Isolation Forest training
│   ├── train5.py                       # Fusion weight grid search
│   └── train_fr.py                     # Zero-day evaluation
│
├── evaluate/
│   ├── metrics.py
│   ├── analyses.py                     # PR curve analysis
│   └── metric2_friday.py               # Binary detection metrics
│
├── datasets/
│   ├── raw/                            # CICIDS2017 CSVs (not in repo)
│   └── processed/                      # Trained models, preprocessed data
│
├── outputs/                            # Figures and result tables
├── README.md
└── LICENSE
```

---

## Implementation Details

**Random Forest** — 100 trees via bootstrap aggregation. Gini impurity splits over random feature subspaces of size √p. `max_depth=10`, `min_samples_split=10`. Parallelised with `ProcessPoolExecutor` (6 workers). Feature importance via sample-weighted Mean Decrease in Impurity. `predict_proba()` returns the fraction of trees voting per class.

**Isolation Forest** — 100 isolation trees, subsample size 256 (following Liu et al., 2008). Anomaly score `s(x) = 2^(−E[h(x)] / c(n))`, where `c(n)` is the average path length of an unsuccessful BST search. Fixed seed (42).

**Weighted fusion**

```python
class WeightedHybrid_IDS:
    def combined_score(self, X):
        p_attack_rf = 1 - self.rf.predict_proba(X)[:, 0]
        s_anomaly_if = self.iforest.score(X)
        return self.w_rf * p_attack_rf + self.w_if * s_anomaly_if

    def predict(self, X):
        return (self.combined_score(X) >= self.threshold).astype(np.int16)
```

---

## Known Limitations

- **Botnet detection is weak** (1.13% recall). Stealth C&C traffic is nearly indistinguishable from benign flows at this feature resolution. This is not operational protection.
- **Feature-representation bound.** The system sees only the 77 CICFlowMeter flow statistics. Signal in packet payloads or session-level behaviour is invisible by construction.
- **Single-day tuning anchor.** Weights and threshold are fitted on Wednesday traffic. Generalisation to other network environments is untested.

---

## Roadmap

- Per-family adaptive weight tuning
- Richer feature representations (packet / session level)
- SHAP per-alert explainability
- Automated PDF alert reports for SOC analysts
- Real-time streaming deployment

---

## Citation & Attribution

**This project is released under a noncommercial license with an attribution requirement.** You're free to use, modify, and distribute this code for research, education, and personal projects — but **not to sell it or use it in a commercial product or service.**

**Required attribution:** any use of this code, in whole or in part, must include visible credit to the original author via a citation, a credit line in your README or documentation, or a link back to this repository.

**Commercial use requires permission.** If you want to use this in a commercial setting, contact me first at [moulaycht@gmail.com](mailto:moulaycht@gmail.com).

### Academic citation

```bibtex
@thesis{driss2026hybrid,
  title  = {Hybrid AI-Powered Intrusion Detection System: Combining
            Supervised and Unsupervised Machine Learning for Zero-Day
            Attack Detection},
  author = {Moulay Driss},
  year   = {2026},
  school = {Faculté des Sciences de Tétouan, Université Abdelmalek Essaâdi},
  type   = {Bachelor's Dissertation}
}
```

Plain text:

> Moulay Driss (2026). *Hybrid AI-Powered Intrusion Detection System: Combining Supervised and Unsupervised Machine Learning for Zero-Day Attack Detection.* Bachelor's Dissertation, Faculté des Sciences de Tétouan, Université Abdelmalek Essaâdi.

**You may** use this for research, teaching, learning, and personal projects; modify and extend it; and redistribute it under the same terms.

**You must** retain the copyright notice, give visible credit to the original author, and not present this work as your own original creation.

**You may not** sell this code, include it in a paid product or service, or use it for any commercial purpose without written permission.

Please don't submit this code — or lightly modified versions of it — as your own academic coursework. This was built as original research, and academic integrity depends on the same standard being upheld by others.

---

## License

Based on the **PolyForm Noncommercial License 1.0.0** ([full text](https://polyformproject.org/licenses/noncommercial/1.0.0/)), with an added attribution requirement.

```
Copyright (c) 2026 Moulay Driss

NONCOMMERCIAL LICENSE WITH ATTRIBUTION REQUIREMENT

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to use,
copy, modify, merge, publish, and distribute the Software, subject to the
following conditions:

1. NONCOMMERCIAL USE ONLY. The Software may be used only for noncommercial
   purposes. Noncommercial purposes include personal use, academic research,
   teaching, and study. Any use of the Software in connection with a product
   or service offered for sale, or otherwise intended for commercial
   advantage or monetary compensation, requires prior written permission from
   the copyright holder.

2. ATTRIBUTION REQUIREMENT. Any use of the Software, in whole or in part,
   including derivative works, must include visible attribution to the
   original author (Moulay Driss) in the accompanying documentation, README,
   published paper, or user interface, along with a link to the original
   repository.

3. NOTICE. The above copyright notice and this permission notice shall be
   included in all copies or substantial portions of the Software.

4. SAME TERMS. Redistributed copies and derivative works must be licensed
   under these same terms.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

For commercial licensing enquiries, contact [moulaycht@gmail.com](mailto:moulaycht@gmail.com).

---

## Author

**Moulay Driss**
Final-year Computer Science student
Faculté des Sciences de Tétouan — Université Abdelmalek Essaâdi

📧 [moulaycht@gmail.com](mailto:moulaycht@gmail.com)

For commercial licensing enquiries, questions about the implementation, or collaboration, feel free to get in touch.

---

<p align="center">
  <em>Built from scratch. Evaluated honestly.</em>
</p>