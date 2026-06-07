<div align="center">

# Fraud Detection Using Financial Transactions

### Machine Learning project for detecting fraudulent online payment transactions

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Best%20Model-FF6600?style=for-the-badge)
![Status](https://img.shields.io/badge/Project-Completed-2EA44F?style=for-the-badge)

</div>

---

## Overview

This project builds a machine learning pipeline to detect fraudulent financial transactions from a large online payments dataset.

Fraud detection is a critical task for financial institutions because fraudulent transactions can cause financial loss, security risk, and loss of customer trust. The main challenge in this project is the severe class imbalance between normal and fraudulent transactions.

The workflow includes data exploration, preprocessing, outlier handling, class balancing, feature selection, model training, and model comparison.

---

## Problem Statement

The goal is to classify each financial transaction as either legitimate or fraudulent.

| Class | Meaning |
|---|---|
| `0` | Non-fraud transaction |
| `1` | Fraud transaction |

Because fraud cases are rare compared to normal transactions, a model trained directly on the original dataset may become biased toward the majority class. To reduce this issue, the project applies undersampling to create a balanced training dataset.

---

## Dataset

The notebook uses the following dataset file:

```text
Online_Payments_Fraud.csv
```

The dataset contains millions of online payment transactions.

### Dataset Size

| Property | Value |
|---|---:|
| Rows | 6,362,620 |
| Columns | 11 |
| Fraud transactions | 8,213 |
| Non-fraud transactions | 6,354,407 |
| Fraud ratio | Approximately 0.129% |

### Dataset Columns

| Column | Description |
|---|---|
| `step` | Time step of the transaction |
| `type` | Transaction type |
| `amount` | Transaction amount |
| `nameOrig` | Sender/customer identifier |
| `oldbalanceOrg` | Sender balance before the transaction |
| `newbalanceOrig` | Sender balance after the transaction |
| `nameDest` | Receiver/customer identifier |
| `oldbalanceDest` | Receiver balance before the transaction |
| `newbalanceDest` | Receiver balance after the transaction |
| `isFraud` | Target label: fraud or non-fraud |
| `isFlaggedFraud` | System-generated fraud flag |

### Transaction Types

| Transaction Type | Count |
|---|---:|
| `CASH_OUT` | 2,237,500 |
| `PAYMENT` | 2,151,495 |
| `CASH_IN` | 1,399,284 |
| `TRANSFER` | 532,909 |
| `DEBIT` | 41,432 |

---

## Data Quality Checks

The notebook performs initial checks before preprocessing.

| Check | Result |
|---|---|
| Missing values | 0 |
| Duplicate rows | 0 |
| Dataset shape | `(6362620, 11)` |
| Memory usage | Approximately 534 MB |

The dataset is clean in terms of missing values and duplicates, but it is highly imbalanced.

---

## Exploratory Data Analysis

The project analyzes the dataset using several visualizations and statistical checks.

Main analysis points:

- Distribution of transaction types.
- Distribution of transaction amounts.
- Fraud vs non-fraud class distribution.
- Frequency of sender and receiver accounts.
- Distribution of the `step` feature.
- Correlation matrix between numerical features.
- Box plot of transaction amount by fraud status.
- Distribution of fraudulent transaction amounts.
- Class distribution before and after undersampling.
- Model performance comparison.

Key observations:

- Fraud transactions are mainly associated with `TRANSFER` and `CASH_OUT`.
- The original dataset contains very few fraud cases compared to normal transactions.
- Some balance-related features are strongly correlated.
- High transaction amounts are more relevant when analyzing fraud behavior.
- The `isFlaggedFraud` feature has very limited positive flags and is not enough by itself for fraud detection.

---

## Preprocessing Pipeline

The preprocessing stage includes the following steps:

### 1. Missing Value Check

All columns were checked for missing values.

Result:

```text
No missing values were found.
```

### 2. Duplicate Check

The dataset was checked for duplicate rows.

Result:

```text
No duplicate rows were found.
```

### 3. Outlier Handling

The notebook applies percentile-based capping to reduce the impact of extreme values.

The following numerical columns were capped using the 10th and 90th percentiles:

```text
amount
oldbalanceOrg
newbalanceOrig
oldbalanceDest
newbalanceDest
```

This step helps reduce extreme skewness while preserving the overall structure of the data.

### 4. Class Balancing

The original dataset is highly imbalanced:

```text
Non-fraud: 6,354,407
Fraud: 8,213
```

To address this, undersampling is applied to the majority class.

After undersampling:

```text
Non-fraud: 8,213
Fraud: 8,213
Total: 16,426
```

### 5. Encoding Transaction Type

The categorical `type` column is converted into numerical values.

| Transaction Type | Encoded Value |
|---|---:|
| `CASH_OUT` | 1 |
| `PAYMENT` | 2 |
| `CASH_IN` | 3 |
| `TRANSFER` | 4 |
| `DEBIT` | 5 |

### 6. Feature Selection

The notebook selects the following input features:

```text
type
amount
oldbalanceOrg
newbalanceOrig
```

The target variable is:

```text
isFraud
```

### 7. Feature Scaling

The selected features are standardized using `StandardScaler`.

Scaling is important because some models, such as Logistic Regression and SVC, are sensitive to feature magnitude.

---

## Machine Learning Workflow

```text
Load Dataset
      |
      v
Explore Dataset
      |
      v
Check Missing Values and Duplicates
      |
      v
Analyze Class Imbalance
      |
      v
Handle Outliers
      |
      v
Balance Classes with Undersampling
      |
      v
Encode Transaction Type
      |
      v
Select Features
      |
      v
Split Data into Train and Test Sets
      |
      v
Apply Standard Scaling
      |
      v
Train Multiple ML Models
      |
      v
Evaluate and Compare Results
```

---

## Train/Test Split

The balanced dataset is split into training and testing sets.

| Setting | Value |
|---|---|
| Training set | 80% |
| Testing set | 20% |
| Random state | 0 |

---

## Models Used

Four classification models are trained and compared.

| Model | Purpose |
|---|---|
| Logistic Regression | Baseline linear classifier |
| XGBoost Classifier | Gradient boosting model for strong tabular performance |
| Gaussian Naive Bayes | Probabilistic baseline classifier |
| Support Vector Classifier | Margin-based classifier for binary classification |

---

## Evaluation Metrics

The models are evaluated using the following metrics:

| Metric | Description |
|---|---|
| Accuracy | Overall percentage of correct predictions |
| Precision | Percentage of predicted fraud cases that were correct |
| Recall | Percentage of actual fraud cases detected |
| F1 Score | Harmonic mean of precision and recall |
| F-Beta Score | Precision-recall metric with beta = 0.5 |

---

## Results

| Model | Train Accuracy | Test Accuracy | Precision | Recall | F1 Score | F-Beta |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 89.91% | 90.47% | 90.47% | 90.47% | 90.47% | 90.47% |
| XGBoost | 99.51% | 99.42% | 99.42% | 99.42% | 99.42% | 99.42% |
| Naive Bayes | 78.19% | 78.45% | 78.45% | 78.45% | 78.45% | 78.45% |
| SVC | 94.98% | 94.77% | 94.77% | 94.77% | 94.77% | 94.77% |

---

## Best Model

The best-performing model is:

```text
XGBoost Classifier
```

Final performance:

| Metric | Value |
|---|---:|
| Test Accuracy | 99.42% |
| Precision | 99.42% |
| Recall | 99.42% |
| F1 Score | 99.42% |
| F-Beta Score | 99.42% |

XGBoost achieved the strongest performance among all tested models.

---

## Key Findings

- The dataset is extremely imbalanced, with fraud representing only about 0.129% of all transactions.
- Fraudulent transactions are mainly found in `TRANSFER` and `CASH_OUT` transaction types.
- The selected features `type`, `amount`, `oldbalanceOrg`, and `newbalanceOrig` were used for model training.
- Undersampling produced a balanced dataset with equal fraud and non-fraud samples.
- XGBoost achieved the highest test accuracy.
- Naive Bayes produced the weakest result among the tested models.
- SVC performed well but is computationally heavier than simpler classifiers.
- The `isFlaggedFraud` column alone is insufficient because it flags only a very small number of transactions.

---

## Repository Structure

```text
fraud-detection-financial-transactions/
│
├── fraud_detection_financial_transactions.ipynb
├── fraud_detection_report.pdf
├── README.md
└── .gitignore
```

---

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/Adnanwadee/fraud-detection-financial-transactions.git
```

### 2. Open the Project Folder

```bash
cd fraud-detection-financial-transactions
```

### 3. Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
```

### 4. Add the Dataset

Place the dataset file in the project directory or update the path inside the notebook.

Expected filename:

```text
Online_Payments_Fraud.csv
```

### 5. Run the Notebook

```bash
jupyter notebook fraud_detection_financial_transactions.ipynb
```

Then run the notebook cells from top to bottom.

---

## Technologies Used

| Category | Tools |
|---|---|
| Programming Language | Python |
| Development Environment | Jupyter Notebook / Google Colab |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, XGBoost |
| Preprocessing | StandardScaler, Label Encoding |
| Evaluation | Accuracy, Precision, Recall, F1 Score, F-Beta Score |

---

## Report

The academic report is included in this repository:

```text
fraud_detection_report.pdf
```

The report covers:

- Problem definition
- Dataset description
- Preprocessing steps
- Exploratory visualizations
- Model selection
- Training and evaluation
- Results and insights
- Project challenges

---

## Limitations

- The dataset is not included in the repository because it is large.
- The project uses undersampling, which reduces the number of non-fraud samples.
- The model is evaluated on a balanced dataset, not the original imbalanced distribution.
- The notebook focuses on experimentation and analysis, not production deployment.
- The final trained model is not exported as a reusable model file.
- No API, dashboard, or real-time fraud detection system is included.

---

## Future Improvements

Possible improvements include:

- Add confusion matrices for every model.
- Add ROC-AUC and PR-AUC evaluation.
- Use SMOTE or hybrid sampling methods.
- Tune XGBoost hyperparameters.
- Add feature importance analysis.
- Export the best model using `joblib` or `pickle`.
- Build a FastAPI endpoint for fraud prediction.
- Build a Streamlit dashboard for interactive analysis.
- Test the final model against the original imbalanced distribution.
- Add explainability using SHAP.

---

## Authors

| Name | Contribution |
|---|---|
| Adnan Wadee Abdullah | Data analysis, preprocessing, modeling, evaluation |
| Ammar Atef Alrousan | Data analysis, preprocessing, modeling, evaluation |

---

## Project Category

```text
Machine Learning
Binary Classification
Fraud Detection
Financial Transactions
Imbalanced Data
```

---

<div align="center">

### Final Summary

This project demonstrates a complete machine learning workflow for detecting fraudulent online payment transactions.  
The pipeline starts from raw transaction analysis, handles data imbalance, trains multiple classifiers, and compares their performance.

**Best Model:** XGBoost Classifier  
**Best Test Accuracy:** 99.42%

</div>
