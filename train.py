from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)
SEED = 42
raw = load_breast_cancer(as_frame=True)
X = raw.data
# Target semantics: 0 = malignant, 1 = benign (the AUC positive class is 1).
y = raw.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y
)
test_csv = X_test.copy()
test_csv["target"] = y_test
test_csv.to_csv(ROOT / "test_data.csv", index=False)

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=SEED),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=SEED),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(C=1.0, kernel="rbf", probability=True, random_state=SEED),
}
slugs = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "KNN": "knn",
    "Naive Bayes": "naive_bayes",
    "Support Vector Machine": "svm",
}
result_rows = []
for name, estimator in models.items():
    pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", estimator)])
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    probability_for_class_1 = pipeline.predict_proba(X_test)[:, list(pipeline.classes_).index(1)]
    result_rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "AUC": roc_auc_score(y_test, probability_for_class_1),
        "Precision": precision_score(y_test, pred, pos_label=1, zero_division=0),
        "Recall": recall_score(y_test, pred, pos_label=1, zero_division=0),
        "F1": f1_score(y_test, pred, pos_label=1, zero_division=0),
        "MCC": matthews_corrcoef(y_test, pred),
    })
    joblib.dump(pipeline, MODEL_DIR / f"{slugs[name]}.joblib")

pd.DataFrame(result_rows).to_csv(ROOT / "metrics.csv", index=False)
with (ROOT / "metadata.json").open("w", encoding="utf-8") as file:
    json.dump({"features": X.columns.tolist(), "target": "target",
               "class_labels": {"0": "malignant", "1": "benign"},
               "dataset_rows": len(X), "dataset_features": X.shape[1],
               "test_rows": len(X_test), "seed": SEED,
               "model_files": {name: f"model/{slug}.joblib" for name, slug in slugs.items()}}, file, indent=2)
print(pd.DataFrame(result_rows).round(4).to_string(index=False))
print(f"Test records: {len(y_test)}; training records: {len(y_train)}")
