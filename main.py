import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

csv_path = "./bank.csv"
df = pd.read_csv(csv_path)


print(df.shape)
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df["deposit"].value_counts())

num_cols = ["age", "balance", "duration", "campaign", "pdays", "previous"]
print("\n Mean values \n")
print(df.groupby("deposit")[num_cols].mean().round(2).T)

print("\n Median values \n")
print(df.groupby("deposit")[num_cols].median().round(2).T)
## We notice that the balance and the duration are significantly higher values where deposit == 'yes'
## Campaign shows that too many client calls lowers the chance of deposit
df["target_numeric"] = (df["deposit"] == "yes").astype(int)

plt.figure(figsize=(18, 14))

cat_cols = [
    "job",
    "marital",
    "education",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
]

for i, col in enumerate(cat_cols, 1):
    plt.subplot(3, 3, i)
    ax = sns.barplot(
        x=col,
        y="target_numeric",
        data=df,
        palette="Blues_d",
        errorbar=None,
        hue=col,
        legend=False,
        order=df.groupby(col)["target_numeric"].mean().sort_values(ascending=False).index,
    )

    plt.title(f"Acceptance rate by: {col}", fontsize=11, fontweight="bold")
    plt.ylabel("% Accepted deposits")
    plt.xlabel("")
    plt.xticks(rotation=45, ha="right")

    # Add percentage labels on top of each bar
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height()*100:.1f}%",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="bottom",
            fontsize=8,
            xytext=(0, 3),
            textcoords="offset points",
        )

plt.tight_layout()
plt.show()


## pdays == -1 tells us that clients haven't been contacted before
bins = [-2, 0, 30, 90, 999]
labels = ["never", "recent_month", "recent_quarter", "old"]

df["pdays_group"] = pd.cut(df["pdays"], bins=bins, labels=labels)
## is_liquid finds ideal client with no credits/loans and positive balance
df['is_liquid'] = ((df['balance'] > 0) & (df['housing'] == 'no') & (df['loan'] == 'no')).astype(int)
## students and retired people have higher chances of saying yes
df['is_high_response_job'] = df['job'].isin(['student', 'retired']).astype(int)
## creating groups for different ages where young and seniors react different to offers
df['age_group'] = pd.cut(df['age'], bins=[0, 30, 60, 100], labels=['young', 'middle', 'senior'])
## total number of times the client was contacted (this campaign + previous campaigns)
df['total_contacts'] = df['campaign'] + df['previous']
## people who accepted the offer at the last campaign and have great balance
df['super_client'] = ((df['poutcome'] == 'success') & (df['balance'] > 1000)).astype(int)

plt.figure(figsize=(10, 6))
sns.heatmap(df[num_cols + ['target_numeric']].corr(), annot=True, cmap='Blues', fmt=".2f")
plt.title("Correlation matrix of numerical features")
plt.show()

plt.figure(figsize=(10, 4))
sns.boxplot(x='deposit', y='duration', data=df, hue='deposit', palette='Blues', legend=False)
plt.title("Effect of call duration on deposit acceptance")
plt.show()

plt.figure(figsize=(12, 4))
sns.countplot(x='month', hue='deposit', data=df, order=['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
plt.title("Total number of calls per month split by outcome")
plt.show()

## note: Duration was removed because it would cause data leakage as an unfamiliar column during actual predictions
## (the call duration is only known after the call ends, and by then the outcome is already known).
## removing it causes model to be usable in an actual data.


num_features = [
    'age', 'balance', 'campaign', 'pdays', 'previous',
    'total_contacts', 'is_liquid', 'is_high_response_job', 'super_client'
]

cat_features = [
    'job', 'marital', 'education', 'housing', 'loan',
    'contact', 'month', 'poutcome', 'pdays_group', 'age_group'
]

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_features),
        ('cat', categorical_transformer, cat_features)
    ])


pipe_lr = Pipeline(
    [
        ("prep", preprocessor),
        ("classifier", LogisticRegression(random_state=42)),
    ]
)

pipe_knn = Pipeline(
    [
        ("prep", preprocessor),
        (
            "classifier",
            KNeighborsClassifier(
                n_neighbors=7, weights="distance", metric="euclidean"
            ),
        ),
    ]
)
pipe_dtc = Pipeline(
    [
        ("prep", preprocessor),
        (
            "classifier",
            DecisionTreeClassifier(
                max_depth=3, min_samples_leaf=5, random_state=42
            ),
        ),
    ]
)
pipe_rfc = Pipeline(
    [
        ("prep", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=50, max_depth=5, random_state=42, n_jobs=-1
            ),
        ),
    ]
)

X = df[num_features + cat_features]
y = df['target_numeric']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# Why Random Forest was chosen for GridSearchCV:
## Random Forest was chosen for tuning because it is a strong candidate for this type of tabular problem
## and can model non-linear relationships and feature interactions.
## Logistic Regression is a linear model that struggles with complex, non-linear relationships.
## K-Nearest Neighbors (K-NN) is computationally expensive and sensitive to high dimensionality
## resulting from One-Hot Encoding.
## While a single Decision Tree tends to overfit, Random Forest averages multiple trees
## to yield higher stability, robustness, and predictive accuracy, making it the ideal
## candidate for intensive hyperparameter optimization via GridSearchCV.
param_grid = {
    "classifier__n_estimators": [50, 100, 200],
    "classifier__max_depth": [None, 10, 20, 30],
    "classifier__min_samples_split": [2, 5, 10],
}

grid_search = GridSearchCV(
    estimator=pipe_rfc,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
)

grid_search.fit(X_train, y_train)

best_rf_model = grid_search.best_estimator_
y_pred_best = best_rf_model.predict(X_test)

print("\n--- RESULTS OF THE BEST RANDOM FOREST MODEL (TUNED) ---")
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best validation F1 score (GridSearch): {grid_search.best_score_:.4f}")

print("\nClassification Report for the best Random Forest:")
print(classification_report(y_test, y_pred_best))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_best, cmap="Blues")
plt.title("Confusion Matrix - Best Tuned Random Forest")
plt.show()


pipe_lr.fit(X_train, y_train)
pipe_knn.fit(X_train, y_train)
pipe_dtc.fit(X_train, y_train)
pipe_rfc.fit(X_train, y_train)

accuracy = pipe_lr.score(X_test, y_test)
print(f"LR model accuracy on the test set: {accuracy * 100:.2f}%")
accuracy = pipe_knn.score(X_test, y_test)
print(f"KNN model accuracy on the test set: {accuracy * 100:.2f}%")
accuracy = pipe_dtc.score(X_test, y_test)
print(f"DTC model accuracy on the test set: {accuracy * 100:.2f}%")
accuracy = pipe_rfc.score(X_test, y_test)
print(f"RFC model accuracy on the test set: {accuracy * 100:.2f}%")

## Class imbalance can cause a problem in data sets, in this case the relationship between yes/no
## is roughly 53% to 47% which causes accuracy to not be much misleading
## in other cases if a model picks data with 88/12 relationship, it can cause false Accuracy.

# Feature importance is taken from the tuned Random Forest (best_rf_model), not the base pipe_rfc
feature_names = best_rf_model.named_steps['prep'].get_feature_names_out()

importances = best_rf_model.named_steps['classifier'].feature_importances_

df_importances = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False).reset_index(drop=True)

print("\n--- The most important columns for the tuned Random Forest ---")
print(df_importances.head(15))

pipelines = {
    "Logistic Regression": pipe_lr,
    "K-NN": pipe_knn,
    "Decision Tree": pipe_dtc,
    "Random Forest": pipe_rfc
}


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_results = []

for name, model_pipe in pipelines.items():
    cv_out = cross_validate(model_pipe, X, y, cv=cv, scoring=['accuracy', 'f1'], n_jobs=-1)

    cv_results.append({
        "Model": name,
        "CV Accuracy (%)": round(cv_out['test_accuracy'].mean() * 100, 2),
        "CV F1 (%)": round(cv_out['test_f1'].mean() * 100, 2)
    })

df_results = pd.DataFrame(cv_results)
df_results = df_results.sort_values(by="CV F1 (%)", ascending=False).reset_index(drop=True)

print("\n--- Comparing models without duration ---")
print(df_results.to_string(index=False))

# Precision / Recall / F1 + Confusion Matrix for every model
for name, p in pipelines.items():
    y_pred = p.predict(X_test)
    print(f"\n--- {name} ---")
    print(classification_report(y_test, y_pred))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.show()
