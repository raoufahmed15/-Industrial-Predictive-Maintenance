# Industrial Predictive Maintenance & Cost-Sensitive Failure Detection

> A machine-learning predictive maintenance project built on the **AI4I 2020 Predictive Maintenance Dataset**, designed around two industrial tasks: **binary machine-failure detection** and **multi-label failure-mode identification**, with explicit handling of class imbalance, decision-threshold tuning, and cost-sensitive evaluation.

**Author:** Raouf Ahmed

**GitHub Repository:** [Industrial Predictive Maintenance](https://github.com/raoufahmed15/-Industrial-Predictive-Maintenance)

**Live Demo:** [Industrial Predictive Maintenance — Streamlit](https://2vovkvkurktczvexxji2li.streamlit.app/)

---

## Table of contents

1. [Project Overview](#project-overview)
2. [What's in the box](#whats-in-the-box)
3. [Dataset](#dataset)
4. [Target Definition](#target-definition)
5. [Industrial EDA](#industrial-eda)
6. [Feature Engineering](#feature-engineering)
7. [Train / Validation / Test Split](#train--validation--test-split)
8. [Model Strategy](#model-strategy)
9. [Threshold Tuning](#threshold-tuning)
10. [Cost-Sensitive Evaluation](#cost-sensitive-evaluation)
11. [Final Evaluation](#final-evaluation)
12. [Failure Mode Identification](#failure-mode-identification)
13. [Explainability](#explainability)
14. [Anomaly Detection](#anomaly-detection)
15. [RUL Feasibility](#rul-feasibility)
16. [Model Artifacts](#model-artifacts)
17. [Project Structure](#project-structure)
18. [Quick Start](#quick-start)
19. [Limitations](#limitations)
20. [Future Improvements](#future-improvements)
21. [Author](#author)

---

## Project Overview

This project addresses predictive maintenance for industrial machines using sensor and operating-condition data from the **AI4I 2020 Predictive Maintenance Dataset**.

The notebook is organized around the idea that an industrial maintenance system should answer two different questions rather than force everything into one classifier:

1. **Task A — Failure Detection:** Will the machine fail?
2. **Task B — Failure Mode Identification:** Which failure mode(s) are present?

Task A is modeled as a **binary classification problem**. Task B is modeled as **five independent binary classification problems using One-vs-Rest**, making it a **multi-label** problem rather than a single multiclass problem.

The project also treats the decision threshold as an industrial control rather than blindly using `0.50`. Validation-only threshold tuning is used to balance missed failures and unnecessary maintenance actions, followed by a final evaluation on a held-out test set.

---

## What's in the box

### Implemented in the notebook

- AI4I 2020 dataset loading and structural audit
- Missing-value and duplicate checks
- Product-ID uniqueness audit
- Target consistency audit between `Machine failure` and failure-mode flags
- Failure-rate and class-imbalance analysis
- Sensor distribution and failure-conditioned EDA
- Product-type failure-rate analysis
- Failure-mode frequency analysis
- Correlation analysis
- Domain-inspired feature engineering
- Stratified train / validation / test splitting
- One-hot encoding for product type
- Standardization of numeric features
- Logistic Regression baseline
- Random Forest baseline
- XGBoost baseline
- ROC-AUC and PR-AUC evaluation
- Precision / Recall / F1 / FDR reporting
- Validation threshold sweep
- Business-rule threshold selection
- Cost-sensitive threshold analysis
- Final held-out test evaluation
- Multi-label failure-mode modeling with One-vs-Rest Random Forest
- Random Forest feature importance
- SHAP-based explainability
- Isolation Forest anomaly detection
- RUL feasibility assessment
- Export of trained models as `.pkl` files with Joblib

---

## Dataset

**Dataset:** AI4I 2020 Predictive Maintenance Dataset (Kaggle)

The notebook loads the dataset from:

```python
pd.read_csv("ai4i2020.csv")
```

### Dataset statistics used in the notebook

- **Rows:** 10,000
- **Columns:** 14
- **Missing values:** 0
- **Duplicate rows:** 0
- **Duplicate Product IDs:** 0
- **Machine failures:** 339
- **Normal samples:** 9,661
- **Failure rate:** 3.39%
- **Random seed:** 42

### Original columns

| Column | Role |
|---|---|
| `UDI` | Row / unit index used by the source dataset |
| `Product ID` | Product identifier |
| `Type` | Product type (`L`, `M`, `H`) |
| `Air temperature [K]` | Air temperature sensor |
| `Process temperature [K]` | Process temperature sensor |
| `Rotational speed [rpm]` | Rotational speed |
| `Torque [Nm]` | Torque |
| `Tool wear [min]` | Tool wear |
| `Machine failure` | Primary binary failure target |
| `TWF` | Tool Wear Failure flag |
| `HDF` | Heat Dissipation Failure flag |
| `PWF` | Power Failure flag |
| `OSF` | Overstrain Failure flag |
| `RNF` | Random Failure flag |

### Target consistency audit

The notebook explicitly checks whether `Machine failure` is simply the logical OR of the five failure-mode columns. It is not.

| Audit item | Count |
|---|---:|
| `Machine failure = 1` | **339** |
| Any failure subtype flagged | **348** |
| Failure with no subtype flag | **9** |
| Subtype flagged while `Machine failure = 0` | **18** |
| Rows with more than one subtype | **24** |

This supports the project's two-task formulation and avoids collapsing the five failure modes into a single multiclass target.

---

## Target Definition

### Task A — Failure Detection

The primary target is:

```text
Machine failure
```

It is modeled as:

```text
0 → Normal operation
1 → Machine failure
```

The class is highly imbalanced, with only **3.39%** positive failure cases. Because of this imbalance, the notebook emphasizes **Precision, Recall, FDR, PR-AUC, and threshold behavior** rather than relying on accuracy alone.

### Task B — Failure Mode Identification

The following five columns are treated as independent binary targets:

```text
TWF  → Tool Wear Failure
HDF  → Heat Dissipation Failure
PWF  → Power Failure
OSF  → Overstrain Failure
RNF  → Random Failure
```

Task B is implemented using:

```text
OneVsRestClassifier
        +
RandomForestClassifier
```

This allows more than one failure mode to be predicted for the same sample.

### Failure-mode frequency

| Failure Mode | Full-dataset positives |
|---|---:|
| HDF | **115** |
| OSF | **98** |
| PWF | **95** |
| TWF | **46** |
| RNF | **19** |

`RNF` is particularly sparse, so its test-set metrics are unstable and should be interpreted cautiously.

---

## Industrial EDA

The notebook investigates how the operating variables behave across normal and failure conditions.

### Failure distribution

```text
Normal   : 9,661
Failure  :   339
Failure rate = 3.39%
```

### Failure rate by product type

| Type | Failure rate | Failures | Total |
|---|---:|---:|---:|
| L | 3.9167% | 235 | 6000 |
| M | 2.7694% | 83 | 2997 |
| H | 2.0937% | 21 | 1003 |

### Raw-sensor correlation with `Machine failure`

| Feature | Correlation |
|---|---:|
| `Torque [Nm]` | **0.191321** |
| `Tool wear [min]` | 0.105448 |
| `Air temperature [K]` | 0.082556 |
| `Rotational speed [rpm]` | -0.044188 |
| `Process temperature [K]` | 0.035946 |

Among the raw sensors, **Torque [Nm]** has the strongest linear correlation with the primary failure target in this analysis.

> Correlation is used here as an exploratory signal, not as proof that a feature is causally responsible for failure.

---

## Feature Engineering

The notebook augments the five raw sensor variables with six engineered features intended to capture thermal and mechanical relationships.

### Engineered features

| Feature | Formula / meaning |
|---|---|
| `temp_diff` | Process temperature − Air temperature |
| `temp_ratio` | Process temperature / Air temperature |
| `mechanical_power` | Torque × rotational speed × `2π / 60` |
| `torque_speed_interaction` | Torque × rotational speed |
| `tool_wear_squared` | Tool wear² |
| `power_per_wear` | Mechanical power / (`Tool wear + 1`) |

The resulting Task A feature set contains:

```text
1 categorical feature
+ 5 raw sensor features
+ 6 engineered features
= 12 input features before preprocessing
```

### Correlation after feature engineering

The strongest absolute correlations with `Machine failure` in the combined raw + engineered set were:

| Feature | Correlation |
|---|---:|
| `Torque [Nm]` | **0.191321** |
| `mechanical_power` | 0.176039 |
| `torque_speed_interaction` | 0.176039 |
| `tool_wear_squared` | 0.133873 |
| `temp_diff` | -0.111676 |
| `temp_ratio` | -0.111347 |
| `Tool wear [min]` | 0.105448 |

---

## Train / Validation / Test Split

Task A uses a stratified split so that the failure ratio remains approximately consistent across datasets.

```text
70% Train
15% Validation
15% Test
```

Actual notebook shapes:

| Split | Samples | Features | Failure rate |
|---|---:|---:|---:|
| Train | 7,000 | 12 | 3.39% |
| Validation | 1,500 | 12 | 3.40% |
| Test | 1,500 | 12 | 3.40% |

`random_state = 42` is used throughout.

### Preprocessing

`Type` is encoded using:

```text
OneHotEncoder(handle_unknown="ignore")
```

All numeric features are standardized with:

```text
StandardScaler()
```

The preprocessing is embedded inside scikit-learn pipelines so that transformations are applied consistently during training and inference.

---

## Model Strategy

The notebook compares three supervised baselines for Task A.

### Logistic Regression

- `class_weight="balanced"`
- `max_iter=1000`
- Used as an interpretable linear reference

### Random Forest

- `n_estimators=300`
- `class_weight="balanced"`
- Parallel training with `n_jobs=-1`

### XGBoost

- `n_estimators=300`
- `max_depth=5`
- `learning_rate=0.05`
- `scale_pos_weight` computed from the training-class imbalance
- `eval_metric="logloss"`

### Validation comparison at threshold 0.50

| Model | ROC-AUC | PR-AUC | FDR | Precision | Recall | F1 | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9228 | 0.4339 | 0.8102 | 0.1898 | 0.8039 | 0.3071 | 0.8767 |
| Random Forest | 0.9817 | **0.8774** | **0.0857** | **0.9143** | 0.6275 | 0.7442 | 0.9853 |
| XGBoost | **0.9843** | 0.8037 | 0.2632 | **0.7368** | **0.8235** | **0.7778** | 0.9840 |

The notebook carries **XGBoost** forward for the threshold-tuning and cost-sensitive evaluation stages. Logistic Regression remains the linear reference model, while Random Forest provides an additional strong tree-based baseline.

---

## Threshold Tuning

A fixed threshold of `0.50` is not treated as the final operating point.

Instead, the notebook evaluates XGBoost over thresholds from:

```text
0.05 → 0.95
```

in increments of `0.05`.

The objective is to understand the operational trade-off between:

- Precision
- Recall
- False Discovery Rate (FDR)
- False positives
- False negatives

### FDR-priority business rule

The notebook uses the following validation rule:

```text
Recall >= 0.60
and
minimize FDR
```

The selected threshold is:

```text
0.95
```

Validation result at this threshold:

```text
Precision = 0.8636
Recall    = 0.7451
FDR       = 0.1364
TP        = 38
FP        = 6
FN        = 13
TN        = 1443
```

This is the threshold used for the final held-out test evaluation.

---

## Cost-Sensitive Evaluation

Predictive maintenance has asymmetric consequences:

- A **false positive** may trigger unnecessary maintenance.
- A **false negative** may allow a real failure to go undetected.

The notebook therefore introduces an explicit illustrative cost model:

```text
COST_FP = 1
COST_FN = 15
```

So a missed failure is assumed to cost approximately **15×** as much as an unnecessary maintenance action.

The total cost is defined as:

```text
Total Cost = FP × COST_FP + FN × COST_FN
```

### Minimum-cost threshold in the validation sweep

The lowest recorded validation cost occurs at:

```text
Threshold = 0.20
Total cost = 90
```

At the same time, the FDR-priority rule selects:

```text
Threshold = 0.95
```

These two thresholds serve different business objectives. The notebook uses the **0.95 FDR-priority threshold** for the final test evaluation.

> The FP/FN cost ratio is an assumption used for the notebook's business-rule experiment; it is not a measured maintenance cost from the dataset.

---

## Final Evaluation

The final XGBoost model is evaluated once on the held-out test set using the validation-selected threshold of `0.95`.

### Test-set results

```text
Precision = 0.923
Recall    = 0.706
F1        = 0.800
FDR       = 0.077
Accuracy  = 0.988
ROC-AUC   = 0.979
PR-AUC    = 0.818
```

### Confusion-matrix counts

```text
TP = 36
FP = 3
FN = 15
TN = 1446
```

The final test evaluation is kept separate from threshold selection so that the held-out test set is not used to choose the operating threshold.

---

## Failure Mode Identification

Task B predicts the five failure-mode flags independently using a One-vs-Rest Random Forest pipeline.

```text
Input features
      │
      ▼
Preprocessing
      │
      ▼
One-vs-Rest Random Forest
      │
      ├── TWF
      ├── HDF
      ├── PWF
      ├── OSF
      └── RNF
```

The model uses:

- `RandomForestClassifier(n_estimators=300)`
- `class_weight="balanced"`
- One-vs-Rest formulation
- 80/20 train/test split for the multi-label stage
- Stratification based on `Machine failure`

### Test-set failure-mode results

| Mode | Test positives | Precision (positive class) | Recall (positive class) | F1 (positive class) |
|---|---:|---:|---:|---:|
| TWF | 10 | 0.00 | 0.00 | 0.00 |
| HDF | 29 | 0.93 | 0.93 | 0.93 |
| PWF | 13 | 1.00 | 1.00 | 1.00 |
| OSF | 16 | 0.92 | 0.75 | 0.83 |
| RNF | 4 | 0.00 | 0.00 | 0.00 |

The very small positive counts for **TWF** and especially **RNF** make those metrics sensitive to the particular test split. They should not be interpreted as stable estimates of real-world performance.

---

## Explainability

The project includes two explainability paths for Task A.

### Random Forest feature importance

The notebook reports the following leading feature importances:

| Feature | Importance |
|---|---:|
| `Rotational speed [rpm]` | 0.156807 |
| `Torque [Nm]` | 0.149033 |
| `mechanical_power` | 0.133574 |
| `torque_speed_interaction` | 0.131728 |
| `tool_wear_squared` | 0.091447 |
| `Tool wear [min]` | 0.090334 |
| `temp_ratio` | 0.072216 |
| `temp_diff` | 0.059511 |
| `power_per_wear` | 0.054171 |
| `Air temperature [K]` | 0.030402 |
| `Process temperature [K]` | 0.020286 |

The notebook also generates a **SHAP summary plot** for the XGBoost model to inspect how transformed features contribute to failure predictions.

> Feature importance and SHAP describe model behavior; they should not be treated as proof of physical causation.

---

## Anomaly Detection

An additional unsupervised layer is implemented with **Isolation Forest**.

### Configuration

- `n_estimators = 300`
- `contamination = 0.034`
- `random_state = 42`
- Trained using **normal training samples only**

The anomaly score is then evaluated against actual validation failures.

### Validation result

```text
Isolation Forest ROC-AUC = 0.824
```

The notebook positions this model as a supporting robustness layer rather than a replacement for the supervised failure detector: it does not see failure labels during training.

---

## RUL Feasibility

The notebook evaluates whether the AI4I 2020 data can support a genuine **Remaining Useful Life (RUL)** regression task.

A true RUL problem generally requires, per machine:

- A machine/unit identifier
- A timestamp or cycle index
- Longitudinal sensor history
- An observed failure event after a degradation trajectory
- A meaningful time-to-failure target

The notebook concludes that this dataset does not provide the required machine-level longitudinal trajectories. `UDI` is a row index rather than a time axis, and `Product ID` does not provide a repeated degradation history leading to failure.

Therefore, true RUL estimation is **not reliably identifiable from this dataset** without introducing an artificial proxy.

### Path forward

A genuine RUL extension should use a dataset with real degradation trajectories, such as **NASA C-MAPSS**, and should be treated as a separate extension of the project.

---

## Model Artifacts

The notebook exports two trained model pipelines with Joblib.

### Task A — Failure Detection

```text
xgb_failure_detection_model.pkl
```

Contains the preprocessing pipeline and trained XGBoost classifier used for binary machine-failure detection.

### Task B — Failure Mode Identification

```text
ovr_failure_mode_model.pkl
```

Contains the preprocessing pipeline and One-vs-Rest Random Forest model for the five failure-mode targets.

---

## Project Structure

A deployment-oriented repository can be organized as:

```text
industrial-predictive-maintenance/
│
├── app.py
├── inference.py
├── requirements.txt
├── README.md
├── industrial_predictive_maintenance.ipynb
│
└── models/
    ├── xgb_failure_detection_model.pkl
    └── ovr_failure_mode_model.pkl
```

The notebook remains the reference for data audit, feature engineering, model development, threshold selection, evaluation, explainability, anomaly detection, and model export.

---

## Quick Start

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_PROJECT_FOLDER>
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

The notebook uses the following main Python packages:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
shap
joblib
```

### 3. Dataset

Place the AI4I dataset file in the notebook's working directory:

```text
ai4i2020.csv
```

### 4. Model files

Place the exported model artifacts under:

```text
models/
├── xgb_failure_detection_model.pkl
└── ovr_failure_mode_model.pkl
```

### 5. Run the notebook

Open the notebook and execute the cells to reproduce the data audit, modeling, threshold analysis, evaluation, explainability, anomaly detection, and model export steps.

### 6. Run a deployment app

For a repository containing a Streamlit interface, the application can be started with:

```bash
streamlit run app.py
```

The application should load the exported model pipelines and use the same preprocessing and feature-engineering logic defined during training.

---

## Limitations

- The dataset is strongly imbalanced, with a **3.39%** machine-failure rate.
- The smallest failure mode, `RNF`, has only **19 positive examples** in the full dataset.
- Task B metrics for sparse failure modes can therefore be unstable.
- Validation threshold selection depends on the chosen business rule.
- The `COST_FP = 1` and `COST_FN = 15` values are assumed costs, not measured industrial maintenance costs.
- The final threshold of `0.95` is optimized for the notebook's FDR-priority rule, not for every possible operating objective.
- The dataset does not contain the longitudinal machine histories needed for a genuine RUL model.
- Correlation and model feature importance describe statistical/model relationships and do not establish physical causality.
- Performance measured on this dataset should not be treated as guaranteed performance on a different factory, sensor system, or operating distribution.

---

## Future Improvements

Potential extensions include:

- Calibrating predicted probabilities for more reliable risk scoring
- Optimizing thresholds from real maintenance-cost data rather than assumed costs
- Cost-sensitive or focal-loss approaches for rare failure modes
- More robust cross-validation for sparse Task B labels
- Dedicated modeling strategies for the underrepresented `TWF` and `RNF` modes
- SHAP-based per-prediction maintenance explanations in the deployed application
- Combining supervised failure detection with anomaly scores into a layered decision system
- Monitoring model drift as operating conditions change
- Building a true RUL model using longitudinal degradation datasets
- Adding real-time sensor-stream inference and maintenance-alerting workflows

---

## Author

**Raouf Ahmed**

**Project:** Industrial Predictive Maintenance & Cost-Sensitive Failure Detection
