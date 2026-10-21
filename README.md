# Bank Marketing – Term Deposit Prediction

A beginner machine learning project that predicts whether a bank client will subscribe to a term deposit (`deposit = yes/no`) after a marketing campaign.

## Dataset

[Bank Marketing Dataset](https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset) from Kaggle – 11,162 clients, 16 features + target.

Class distribution: **53% no / 47% yes**, so the dataset is roughly balanced.

## What the project does

1. **Data exploration** – shape, types, missing values, target distribution
2. **EDA** – compares numerical features between yes/no clients and plots the acceptance rate for each categorical feature
3. **pdays handling** – `pdays = -1` means the client was never contacted before, so instead of replacing it with 0 it is grouped into `never / recent_month / recent_quarter / old`
4. **Preprocessing** – `ColumnTransformer` with a numerical pipeline (`SimpleImputer` + `StandardScaler`) and a categorical pipeline (`SimpleImputer` + `OneHotEncoder`)
5. **Models** – Logistic Regression (baseline), KNN, Decision Tree, Random Forest
6. **Evaluation** – Accuracy, Precision, Recall, F1 and Confusion Matrix
7. **Cross-validation** – `StratifiedKFold` + `cross_validate`
8. **Feature engineering** – `pdays_group`, `age_group`, `is_liquid`, `is_high_response_job`, `total_contacts`, `super_client`
9. **Hyperparameter tuning** – `GridSearchCV` on Random Forest (36 combinations, F1 scoring)

### Why `duration` was removed

`duration` is the length of the last call with the client. It is only known **after** the call is finished, and by then the outcome is already known. Using it would be data leakage, so it is left out of the model.

## Results (without `duration`)

| Model               | CV Accuracy | CV F1  |
|---------------------|-------------|--------|
| Logistic Regression | 70.86%      | 65.41% |
| Random Forest       | 70.44%      | 63.44% |
| K-NN                | 67.34%      | 63.12% |
| Decision Tree       | 67.09%      | 62.99% |

**Tuned Random Forest** (test set): Accuracy 73%, Precision (yes) 0.76, Recall (yes) 0.61, F1 (yes) 0.68

## How to run

```bash
git clone <repo-url>
cd bank-marketing-ml
pip install -r requirements.txt
python main.py
```

`bank.csv` must be in the same folder as `main.py`.

## Tech

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn
