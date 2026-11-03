<div align="center">

# 📊 Model Comparison Results

*FloodCast BD — Jamuna River Basin, Sirajganj (2000-2025)*

</div>

---

## 🧪 Experiment Setup

- **Target**: Binary flood occurrence (river discharge > danger level threshold)
- **Validation**: Time-series cross-validation (no future data leakage)
- **Metrics**: Accuracy, F1-Score, AUC-ROC, Precision, Recall
- **Tracking**: All experiments logged in MLflow

---

## 📈 Results Summary

> ⚠️ **Note**: Results will be populated after training with real ERA5/CHIRPS/GloFAS data. Current pipeline is validated with synthetic test fixtures.

### Feature Set A — Rainfall Only

| Model | Accuracy | F1 | AUC | Precision | Recall |
|:---|:---:|:---:|:---:|:---:|:---:|
| XGBoost | — | — | — | — | — |
| LSTM | — | — | — | — | — |
| LSTM + Physics | — | — | — | — | — |

### Feature Set B — Rainfall + Discharge

| Model | Accuracy | F1 | AUC | Precision | Recall |
|:---|:---:|:---:|:---:|:---:|:---:|
| XGBoost | — | — | — | — | — |
| LSTM | — | — | — | — | — |
| LSTM + Physics | — | — | — | — | — |

### Feature Set C — Full Features

| Model | Accuracy | F1 | AUC | Precision | Recall |
|:---|:---:|:---:|:---:|:---:|:---:|
| XGBoost | — | — | — | — | — |
| LSTM | — | — | — | — | — |
| LSTM + Physics | — | — | — | — | — |

---

## 🏆 Best Configuration

| Metric | Best Model | Feature Set | Score |
|:---|:---|:---|:---:|
| Highest F1 | — | — | — |
| Highest AUC | — | — | — |
| Highest Recall | — | — | — |

---

## 🔬 Physics Constraint Impact

| Comparison | F1 Δ | AUC Δ | Recall Δ |
|:---|:---:|:---:|:---:|
| LSTM vs LSTM+Physics (Set A) | — | — | — |
| LSTM vs LSTM+Physics (Set B) | — | — | — |
| LSTM vs LSTM+Physics (Set C) | — | — | — |

> Positive Δ means physics constraints improved the metric.

---

## 📊 Feature Set Ablation

| Model | Set A → B (Δ F1) | Set B → C (Δ F1) |
|:---|:---:|:---:|
| XGBoost | — | — |
| LSTM | — | — |
| LSTM + Physics | — | — |

---

## ⏱️ Training Time (CPU)

| Model | Set A | Set B | Set C |
|:---|:---:|:---:|:---:|
| XGBoost | — | — | — |
| LSTM | — | — | — |
| LSTM + Physics | — | — | — |

---

## 📝 Notes

- All training on CPU (no GPU)
- Hyperparameters tuned with Optuna (50 trials per model)
- Time-series CV ensures no temporal data leakage
- Physics constraints add ~10-15% training time overhead to LSTM
- Results reported as mean ± std across CV folds

---

*Results will be updated after real data training. Check MLflow UI (`mlflow ui`) for detailed experiment logs.*