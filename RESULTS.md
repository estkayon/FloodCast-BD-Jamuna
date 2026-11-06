<div align="center">

# Model Comparison Results

*FloodCast BD — Jamuna River Basin, Sirajganj (2000-2025)*

[![Experiments](https://img.shields.io/badge/Experiments-27-F97316?style=flat-square)](.)
[![Models](https://img.shields.io/badge/Models-3-3B82F6?style=flat-square)](.)
[![Lead Times](https://img.shields.io/badge/Lead%20Times-1d%20·%202d%20·%203d-22C55E?style=flat-square)](.)

</div>

---

## Experiment Setup

| Parameter | Value |
|:---|:---|
| **Target** | Binary flood label (runoff > 90th percentile) |
| **Training Period** | 2000-2022 |
| **Holdout Period** | 2023-2025 |
| **Models** | XGBoost, LSTM, LSTM + Physics |
| **Feature Sets** | Set A (Basic), Set B (Extended), Set C (Physics) |
| **Lead Times** | 1 day, 2 days, 3 days |
| **Total Experiments** | 27 (3 x 3 x 3) |
| **Tracking** | MLflow |

---

## Full Results (27 Experiments)

### 1-Day Lead Time

| Model | Features | AUC | F1 | Precision | Recall |
|:---|:---|:---:|:---:|:---:|:---:|
| XGBoost | Basic (A) | — | — | — | — |
| XGBoost | Extended (B) | — | — | — | — |
| XGBoost | Physics (C) | — | — | — | — |
| LSTM | Basic (A) | — | — | — | — |
| LSTM | Extended (B) | — | — | — | — |
| LSTM | Physics (C) | — | — | — | — |
| LSTM + Physics | Basic (A) | — | — | — | — |
| LSTM + Physics | Extended (B) | — | — | — | — |
| LSTM + Physics | Physics (C) | — | — | — | — |

### 2-Day Lead Time

| Model | Features | AUC | F1 | Precision | Recall |
|:---|:---|:---:|:---:|:---:|:---:|
| XGBoost | Basic (A) | — | — | — | — |
| XGBoost | Extended (B) | — | — | — | — |
| XGBoost | Physics (C) | — | — | — | — |
| LSTM | Basic (A) | — | — | — | — |
| LSTM | Extended (B) | — | — | — | — |
| LSTM | Physics (C) | — | — | — | — |
| LSTM + Physics | Basic (A) | — | — | — | — |
| LSTM + Physics | Extended (B) | — | — | — | — |
| LSTM + Physics | Physics (C) | — | — | — | — |

### 3-Day Lead Time

| Model | Features | AUC | F1 | Precision | Recall |
|:---|:---|:---:|:---:|:---:|:---:|
| XGBoost | Basic (A) | — | — | — | — |
| XGBoost | Extended (B) | — | — | — | — |
| XGBoost | Physics (C) | — | — | — | — |
| LSTM | Basic (A) | — | — | — | — |
| LSTM | Extended (B) | — | — | — | — |
| LSTM | Physics (C) | — | — | — | — |
| LSTM + Physics | Basic (A) | — | — | — | — |
| LSTM + Physics | Extended (B) | — | — | — | — |
| LSTM + Physics | Physics (C) | — | — | — | — |

> **Note**: Fill these tables from `results/experiment_results.csv` after training. Run `python -m floodcast.models.train` to generate results.

---

## Best Configuration

| Metric | Best Model | Feature Set | Lead Time | Score |
|:---|:---|:---|:---:|:---:|
| Highest AUC | — | — | — | — |
| Highest F1 | — | — | — | — |
| Highest Recall | — | — | — | — |

---

## Research Questions

### RQ1: Physics Constraint Impact

Does adding physics constraints to LSTM improve flood prediction?

| Comparison | AUC Delta | F1 Delta | Recall Delta |
|:---|:---:|:---:|:---:|
| LSTM vs LSTM+Physics (Set A, 1d) | — | — | — |
| LSTM vs LSTM+Physics (Set B, 1d) | — | — | — |
| LSTM vs LSTM+Physics (Set C, 1d) | — | — | — |

> Positive delta = physics constraints improved the metric.

### RQ2: Feature Set Ablation

Which features contribute most to prediction accuracy?

| Model | A to B (AUC Delta) | B to C (AUC Delta) |
|:---|:---:|:---:|
| XGBoost (1d) | — | — |
| LSTM (1d) | — | — |
| LSTM + Physics (1d) | — | — |

### RQ3: Lead Time Degradation

How does prediction accuracy degrade with longer lead times?

| Model | 1d AUC | 2d AUC | 3d AUC | Drop (1d to 3d) |
|:---|:---:|:---:|:---:|:---:|
| XGBoost (best set) | — | — | — | — |
| LSTM (best set) | — | — | — | — |
| LSTM + Physics (best set) | — | — | — | — |

---

## Notes

- All training on CPU (no GPU required)
- Holdout validation on 2023-2025 (no temporal data leakage)
- Physics constraints add ~10-15% training time overhead to LSTM
- Check MLflow UI (`mlflow ui`) for detailed experiment logs and artifacts

---

*Last updated: Run `python -m floodcast.models.train` and update this file with real metrics from `results/experiment_results.csv`.*
