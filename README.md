# Breast Tumor Classification - Model Comparison (ML Assignment 2)

## a. Problem statement

Breast tumors can be classified as **malignant (cancerous)** or **benign (non-cancerous)** based on measured characteristics of cell nuclei. The goal of this project is to predict the tumor class using features extracted from digitized images of breast tissue samples.

This is a **binary classification** problem. Five machine learning classifiers are trained on the same dataset and evaluated using **accuracy, AUC, precision, recall, F1 score, and Matthews Correlation Coefficient (MCC)**. A Streamlit web application allows users to upload held-out test data, evaluate individual models, compare all five models, and inspect their ROC curves.

The project is an educational machine learning exercise and **is not intended for medical diagnosis**.

## b. Dataset description

- **Source:** UCI Machine Learning Repository - [Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic)
- **Dataset loaded through:** `sklearn.datasets.load_breast_cancer()` (scikit-learn's copy of the UCI dataset)
- **Total records:** 569
- **Input features:** 30 numerical features
- **Missing values:** None in the dataset used
- **Target column:** `target`
  - `0` = **Malignant** (212 records in the full dataset)
  - `1` = **Benign** (357 records in the full dataset)
- **Classification type:** Binary classification

The 30 features describe measurements of cell nuclei, grouped into three sets of ten:

| Feature group | Description | Number of features |
|---|---|---:|
| Mean | Mean values of radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension | 10 |
| Standard error | Standard-error measurements of the same ten properties | 10 |
| Worst | Largest/worst values of the same ten properties | 10 |

**Preprocessing:** Each classifier is saved as a scikit-learn `Pipeline` that applies `StandardScaler` and then the classification algorithm. The scaler is fitted only on the training data, which prevents leakage from the test set. The model pipelines can therefore accept the original numerical feature columns directly.

**Train/test split:** An 80/20 stratified split with `random_state=42` was used:

| Split | Records | Malignant (0) | Benign (1) |
|---|---:|---:|---:|
| Training | 455 | 170 | 285 |
| Testing | 114 | 42 | 72 |
| **Total** | **569** | **212** | **357** |

The **114 held-out test records** were saved as `test_data.csv`, including all 30 input features and the `target` column. The same test records were used for evaluating all five trained classifiers.

## c. GitHub repository link

**GitHub repository:** https://github.com/VarahaSandeep/breast-tumor-ml-classification

**Live Streamlit application:** https://breast-tumor-ml-classification-zac9boy6yzvtwenc4ahfxk.streamlit.app/

### Repository structure

```text
breast-tumor-ml-classification/
├── app.py                  # Streamlit application and evaluation UI
├── train.py                # Trains and saves the five model pipelines
├── requirements.txt        # Dependencies for local use and cloud deployment
├── README.md               # Project documentation
├── test_data.csv           # Held-out test data with features and target
├── metrics.csv             # Evaluation results for all classifiers
├── metadata.json           # Required features, target, labels and model paths
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── svm.joblib
```

A `.streamlit/config.toml` theme file may also be included if the chosen theme is committed.

The trained models and preprocessing steps are stored together in `.joblib` pipelines. `app.py` **loads** these pipelines and does **not retrain** them on uploaded test data.

## d. Models used

All models were trained using the same 455-row training split and evaluated on the same 114-row held-out test split.

| Classification model | Implementation and settings |
|---|---|
| Logistic Regression | `LogisticRegression(max_iter=2000, random_state=42)` |
| Decision Tree | `DecisionTreeClassifier(max_depth=5, random_state=42)` |
| K-Nearest Neighbors (KNN) | `KNeighborsClassifier(n_neighbors=5)` |
| Gaussian Naive Bayes | `GaussianNB()` |
| Support Vector Machine (SVM) | `SVC(C=1.0, kernel="rbf", probability=True, random_state=42)` |

**Metric definitions:** Accuracy measures overall correct predictions. AUC measures class ranking performance across decision thresholds. Precision measures the reliability of positive predictions; recall measures how many actual positive cases are identified. F1 combines precision and recall, and MCC captures correlation between the true and predicted classes.

For **precision, recall, F1, and AUC**, the positive class in these results is **benign (`1`)**, consistent with the dataset's target encoding. Higher values are better for all the reported metrics; MCC ranges from -1 to +1.

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9211 | 0.9163 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |
| KNN | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| Support Vector Machine | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

*Values are rounded to four decimal places from `metrics.csv`.*

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | Achieved **98.25% accuracy**, **0.9954 AUC**, and **0.9623 MCC**. It tied with SVM on accuracy, precision, recall, F1, and MCC while recording the highest AUC. This suggests the scaled features are highly effective for a linear decision boundary on this test split. |
| **Decision Tree** | Achieved **92.11% accuracy**, the lowest among the five models, with **0.9163 AUC** and **0.8341 MCC**. The depth-limited tree is easier to interpret but makes more classification mistakes than the top-performing models on this split. |
| **KNN** | Achieved **95.61% accuracy**, **0.9722 recall**, and **0.9054 MCC**. This indicates good performance using distances between standardized feature vectors. Its results are strong but remain below Logistic Regression and SVM on accuracy and MCC. |
| **Naive Bayes** | Achieved **92.98% accuracy** and **0.9868 AUC**. Although its overall accuracy and MCC are lower, the high AUC indicates useful ranking performance. The Gaussian independence assumptions may not fully capture relationships among the tumor measurements. |
| **Support Vector Machine** | Achieved **98.25% accuracy**, **0.9950 AUC**, and **0.9623 MCC**. Its RBF kernel matches Logistic Regression on five metrics; its AUC is only slightly lower. It is a close alternative for this dataset. |
| **Overall, winner for this dataset?** | **Logistic Regression**, because it ties SVM on accuracy, precision, recall, F1, and MCC but has the highest AUC (**0.9954** versus **0.9950**). This selection applies to the chosen held-out test split and model settings; it is not a claim of universal superiority. |

## Streamlit application features

The application is organized using left-sidebar navigation:

1. **Upload Data:** Upload and validate `test_data.csv`, inspect sample records, and view the malignant/benign class counts.
2. **Individual Model:** Select a classifier and view six evaluation metrics, a confusion matrix, an aligned classification report, prediction breakdown, and downloadable predictions.
3. **Compare All 5 Models:** Compare Accuracy, AUC, Precision, Recall, F1, and MCC for all five models on the uploaded test set, visualize a metric, and download the comparison.
4. **ROC Curves:** Inspect ROC curves and AUC scores for all five models.

Uploaded data must contain the 30 expected numerical feature columns and a binary `target` column. Invalid formats are rejected. **No model fitting or training is performed in the Streamlit application.**

## How to run locally

**Python 3.12 is recommended** for the dependencies in `requirements.txt`.

```bash
python -m pip install -r requirements.txt
python train.py                     # Optional: regenerate the saved models and test data
python -m streamlit run app.py
```

Open the local address displayed by Streamlit (usually `http://localhost:8501`). Use **Upload Data** to upload `test_data.csv`, then navigate through the three evaluation sections.

## Cloud deployment

The application is deployed through [Streamlit Community Cloud](https://share.streamlit.io/) using the GitHub repository, the `main` branch, and `app.py` as the entry point. The project dependencies are listed in `requirements.txt`; the app loads the saved `.joblib` pipelines from `model/`.

**Application URL:** https://breast-tumor-ml-classification-zac9boy6yzvtwenc4ahfxk.streamlit.app/


