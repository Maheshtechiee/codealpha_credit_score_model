# Credit Score Model 💳

An end-to-end ML pipeline that predicts loan applicant creditworthiness using **Logistic Regression**, **Decision Tree**, and **Random Forest** — with a full diagnostic dashboard.

## Project Structure

```
credit_score_model/
├── main.py                  ← Entry point — runs the full pipeline
├── src/
│   ├── data_generator.py    ← Synthetic credit dataset generation
│   ├── features.py          ← Domain-driven feature engineering
│   ├── preprocessing.py     ← Imputation, train/test split, scaling
│   ├── train.py             ← Model training with 5-fold CV
│   ├── evaluate.py          ← Metrics: Accuracy, F1, ROC-AUC
│   ├── visualize.py         ← 6-panel diagnostic dashboard (PNG)
│   └── predict.py           ← Single applicant prediction demo
├── requirements.txt
└── README.md
```

## Quickstart

```bash
pip install -r requirements.txt
python main.py
```

## Models & Results

| Model               | ROC-AUC | F1-Score |
|---------------------|---------|----------|
| Random Forest       | ~0.91   | ~0.84    |
| Logistic Regression | ~0.87   | ~0.80    |
| Decision Tree       | ~0.82   | ~0.76    |

## Tech Stack

| Layer        | Technology                            |
|--------------|---------------------------------------|
| Language     | Python 3.10+                          |
| ML           | scikit-learn (LR, DT, RF)             |
| Data         | NumPy, Pandas                         |
| Visualization| Matplotlib, Seaborn                   |
| Features     | Custom domain-driven engineering      |

## Key Features

- Synthetic dataset with realistic bureau-style features
- Missing value injection + median imputation
- Feature engineering: risk composite, payment discipline, debt burden score
- 5-fold cross-validation during training
- 6-panel visual dashboard: ROC, PR, confusion matrix, feature importance
- Single applicant prediction demo
