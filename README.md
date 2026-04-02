# Student Dropout Prediction

## Overview

The goal of this project is to predict student dropout based on demographic, academic, and lifestyle-related factors. The analysis includes data cleaning, preprocessing, exploratory data analysis, and comparison of classification models. The main objective is to identify which model performs best at detecting students at risk of dropping out.

---

## Dataset

- **Source:** [Kaggle — Student Dropout Prediction Dataset](https://www.kaggle.com/datasets/meharshanali/student-dropout-prediction-dataset)
- **Size:** 10,000 students, 19 features
- **Target variable:** `Dropout` (0 = no dropout, 1 = dropout)
- **Class distribution:** 76.5% non-dropout / 23.5% dropout

### Features

| Feature | Type | Description |
|---|---|---|
| Age | Continuous | Student age |
| Gender | Binary | Male / Female |
| Family_Income | Continuous | Annual family income |
| Internet_Access | Binary | Access to internet (Yes/No) |
| Study_Hours_per_Day | Continuous | Daily study hours |
| Attendance_Rate | Continuous | Attendance percentage |
| Assignment_Delay_Days | Continuous | Average assignment delay in days |
| Travel_Time_Minutes | Continuous | Daily commute time |
| Part_Time_Job | Binary | Has part-time job (Yes/No) |
| Scholarship | Binary | Receives scholarship (Yes/No) |
| Stress_Index | Continuous | Self-reported stress level (1–10) |
| GPA | Continuous | Grade Point Average |
| Semester | Ordinal | Year 1–4 |
| Department | Categorical | Arts / Business / CS / Engineering / Science |
| Parental_Education | Categorical | High School / Bachelor / Master / PhD / Unknown |

---

## Preprocessing Pipeline

### 1. Missing Value Imputation
- `Family_Income`, `Study_Hours_per_Day`, `Stress_Index` — imputed using **KNNImputer** (n_neighbors=5), preserving relationships between variables
- `Parental_Education` — missing values filled with `"Unknown"` category before encoding

### 2. Feature Encoding
- Binary columns (`Internet_Access`, `Part_Time_Job`, `Scholarship`) — mapped to 0/1 using a loop
- `Gender` — mapped to 0/1
- `Semester` — ordinal mapping (Year 1–4 → 1–4)
- `Department` — One-Hot Encoding
- `Parental_Education` — One-Hot Encoding (avoids ordinal bias; `Unknown` treated as its own category)

### 3. Feature Removal
- `Student_ID` — identifier, not a predictor
- `Semester_GPA`, `CGPA` — removed due to near-perfect correlation with `GPA` (multicollinearity)

### 4. Train-Test Split
- 80/20 split with stratification on target variable

### 5. Feature Scaling
- **StandardScaler** applied exclusively to continuous variables:
  `Age`, `Family_Income`, `Study_Hours_per_Day`, `Attendance_Rate`, `Assignment_Delay_Days`, `Travel_Time_Minutes`, `Stress_Index`, `GPA`
- Binary, ordinal, and OHE columns excluded from scaling

### 6. Class Imbalance Handling
- **SMOTENC** applied to training data — correctly handles mixed data types by interpolating continuous features while randomly sampling binary/categorical ones
- Standard SMOTE was tested and rejected as it generated invalid non-binary values for binary columns

---

## Models

### Logistic Regression (with SMOTENC)
- Trained on SMOTENC-balanced training set
- Evaluated on original (unbalanced) test set

### Logistic Regression (without SMOTE)
- Trained on scaled training set without oversampling
- Included as a baseline comparison to assess SMOTENC impact

### XGBoost (basic)
- Trained on original (unscaled) training set
- Class imbalance handled via `scale_pos_weight=3.25`
- Tree-based models do not require feature scaling

---

## Results

### Logistic Regression — comparison with and without SMOTENC (class 1 — dropout)

| Metric | Without SMOTE | With SMOTENC |
|---|---|---|
| Precision | 0.68 | 0.50 |
| Recall | 0.40 | 0.66 |
| F1-score | 0.51 | 0.57 |
| Accuracy | 0.81 | 0.76 |
| ROC AUC | 0.820 | 0.804 |

### Final model comparison — test set (class 1 — dropout)

| Metric | LR with SMOTENC | XGBoost (basic) |
|---|---|---|
| Precision | 0.50 | 0.52 |
| Recall | 0.66 | 0.54 |
| F1-score | 0.57 | 0.53 |
| Accuracy | 0.76 | 0.77 |
| ROC AUC | 0.804 | 0.776 |
| Train ROC AUC | 0.878 | 0.999 |

### Overfitting assessment

| Model | Train ROC AUC | Test ROC AUC | Gap |
|---|---|---|---|
| Logistic Regression (SMOTENC) | 0.878 | 0.804 | 0.074 |
| XGBoost (basic) | 0.999 | 0.776 | 0.223 |

XGBoost exhibits severe overfitting — it memorised the training set rather than learning generalisable patterns. This explains why a simpler linear model outperforms it on unseen data.

---

## Conclusion

**Logistic Regression with SMOTENC is the recommended model.**

In the context of dropout prediction, recall is the most critical metric — a missed at-risk student means no intervention. Logistic Regression achieves recall of 0.66 versus 0.54 for XGBoost, meaning it detects significantly more students at risk. The absence of severe overfitting (train/test AUC gap of 0.074 vs 0.223 for XGBoost) further confirms its reliability.

SMOTENC improved recall from 0.40 to 0.66 compared to training without oversampling, at the cost of a modest reduction in precision (0.68 → 0.50). This trade-off is justified in the educational context where the cost of missing an at-risk student far outweighs the cost of an unnecessary intervention.

### Note on initial results
The original pipeline used median imputation, standard SMOTE on all features, StandardScaler on all columns, and ordinal encoding for `Parental_Education`. Under those conditions, Logistic Regression achieved 0.47 precision, 0.76 recall, and 0.58 F1-score. The higher recall was partly an artifact of corrupted synthetic data generated by applying SMOTE to binary columns. The updated results are methodologically correct and more reliable.

---

## Project Structure

```
Student Dropout/
├── dropout.ipynb      # Main analysis notebook
├── README.md          # Project documentation
└── requirements.txt   # Python dependencies
```

---

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

Key libraries: `pandas`, `numpy`, `scikit-learn`, `imbalanced-learn`, `xgboost`, `matplotlib`, `seaborn`, `kagglehub`
