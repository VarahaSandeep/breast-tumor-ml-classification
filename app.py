"""Interactive evaluation dashboard for five PRE-TRAINED classifiers.

Upload a held-out CSV with 30 Wisconsin Diagnostic feature columns and target.
All model comparisons run on the same uploaded test data, without retraining.
"""
from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parent
st.set_page_config(
    page_title="Diagnostic ML | Model Lab",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Only presentation is customized: saved models, label meaning, and evaluation
# procedure are unchanged.
st.markdown("""
<style>
    :root {--indigo:#4f46e5;--cyan:#0891b2;--ink:#17233d;--muted:#62718b;}
    .stApp {background:linear-gradient(150deg,#ffffff 0%,#f8fbff 55%,#eef4fb 100%);}
    .block-container {padding-top:1.4rem;padding-bottom:2.5rem;max-width:1450px;}
    [data-testid="stSidebar"] {background:#eef4fb; border-right:1px solid #d6e2f0;}
    [data-testid="stSidebar"] [role="radiogroup"] label {padding:.42rem .48rem;border-radius:9px;}
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {background:#e1e7fd;}
    .hero {background:linear-gradient(112deg,#172554 0%,#3730a3 57%,#0e7490 100%);
       padding:1.6rem 1.75rem;border-radius:20px;box-shadow:0 14px 34px rgba(30,41,59,.15);
       color:white;margin-bottom:1.15rem;}
    .hero .eyebrow {font-size:.82rem;letter-spacing:.16em;text-transform:uppercase;
       font-weight:700;color:#bde9fa;margin:0 0 .5rem;}
    .hero h1 {color:white;font-size:clamp(1.7rem,3vw,2.5rem);line-height:1.15;
       margin:0 0 .5rem;font-weight:780;}
    .hero p {color:#e4edff;margin:0;line-height:1.65;font-size:.98rem;}
    .hero-chip {display:inline-block;border:1px solid rgba(255,255,255,.32);
       background:rgba(255,255,255,.11);color:white;border-radius:99px;
       padding:.32rem .7rem;margin:.85rem .4rem 0 0;font-size:.78rem;font-weight:650;}
    .eyebrow-dark {font-size:.75rem;color:#64748b;font-weight:700;text-transform:uppercase;
       letter-spacing:.12em;margin-bottom:.24rem;}
    .section-title {font-size:1.35rem;font-weight:760;color:#172554;margin:1.25rem 0 .15rem;}
    .section-desc {color:#64748b;font-size:.89rem;margin-bottom:1.05rem;}
    .metric-card {position:relative;min-height:132px;padding:1.12rem 1.15rem;
       margin-bottom:.82rem;background:white;border:1px solid #dce5f8;border-radius:17px;
       box-shadow:0 7px 22px rgba(33,52,93,.05);overflow:hidden;}
    .metric-card:before {content:'';position:absolute;left:0;top:0;bottom:0;width:5px;
       background:var(--accent,#4f46e5);}
    .metric-card .label {font-size:.89rem;font-weight:650;color:#5f6d87;}
    .metric-card .value {font-size:2rem;font-weight:800;color:#192a4b;letter-spacing:-.055rem;
       margin:.4rem 0 .18rem;line-height:1.12;}
    .metric-card .hint {font-size:.77rem;color:#7c8ba3;}
    .mini-stat {background:#fff;border:1px solid #dce5f8;border-radius:14px;
       padding:.85rem 1rem;min-height:80px;}
    .mini-stat .lbl{color:#60708c;font-size:.8rem;font-weight:620;}
    .mini-stat .num{color:#1e3a8a;font-weight:760;font-size:1.42rem;margin-top:.2rem;}
    .data-note {font-size:.83rem;color:#63748f;padding:.7rem 0;}
    div[data-testid="stTabs"] button {font-weight:650;}
    div[data-testid="stDataFrame"] {border-radius:12px;overflow:hidden;}
    .footer-note {color:#66768e;font-size:.83rem;line-height:1.6;margin-top:1.2rem;}

    /* Force an accessible light interface even when the OS/browser prefers dark mode. */
    :root, .stApp, [data-testid="stAppViewContainer"] {
        color-scheme: light !important;
        --text-color: #1b2a3a !important;
        --background-color: #ffffff !important;
        --secondary-background-color: #eef4fb !important;
        --primary-color: #1f6feb !important;
    }
    html, body, .stApp, [data-testid="stAppViewContainer"],
    [data-testid="stMain"], [data-testid="stMainBlockContainer"] {
        color: #17233d !important;
    }
    /* Streamlit native headings, markdown, helper text, and descriptions. */
    [data-testid="stMain"] h1, [data-testid="stMain"] h2,
    [data-testid="stMain"] h3, [data-testid="stMain"] h4,
    [data-testid="stMain"] p, [data-testid="stMain"] li,
    [data-testid="stMain"] label, [data-testid="stMain"] span,
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] {
        color: #17233d;
    }
    [data-testid="stMain"] .hero h1, [data-testid="stMain"] .hero .hero-chip {
        color: #ffffff !important;
    }
    [data-testid="stMain"] .hero p {color:#e4edff !important;}
    [data-testid="stMain"] .hero .eyebrow {color:#bde9fa !important;}
    [data-testid="stMain"] .section-title {color:#172554 !important;}
    [data-testid="stMain"] .section-desc, [data-testid="stMain"] .data-note,
    [data-testid="stMain"] .footer-note, [data-testid="stMain"] .eyebrow-dark {
        color:#475569 !important;
    }
    [data-testid="stMain"] .metric-card .label {color:#475569 !important;}
    [data-testid="stMain"] .metric-card .value {color:#192a4b !important;}
    [data-testid="stMain"] .metric-card .hint {color:#52627b !important;}
    [data-testid="stMain"] .mini-stat .lbl {color:#475569 !important;}
    [data-testid="stMain"] .mini-stat .num {color:#1e3a8a !important;}

    /* Sidebar is lavender, so EVERY native Streamlit label must be dark. */
    [data-testid="stSidebar"], [data-testid="stSidebarContent"],
    [data-testid="stSidebarUserContent"] {
        background-color:#eef4fb !important;
        color:#17233d !important;
        color-scheme:light !important;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] small,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
        color:#17233d !important;
    }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] *,
    [data-testid="stSidebar"] .stCaption {color:#475569 !important;}
    [data-testid="stSidebar"] [role="radiogroup"] label,
    [data-testid="stSidebar"] [role="radiogroup"] label * {
        color:#17233d !important;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background:#dfe7ff !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] > div,
    [data-testid="stSidebar"] [data-baseweb="select"] input,
    [data-testid="stMain"] [data-baseweb="select"] > div,
    [data-testid="stMain"] [data-baseweb="select"] input {
        background-color:#ffffff !important;
        color:#17233d !important;
        -webkit-text-fill-color:#17233d !important;
        border-color:#b9c6e4 !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] *,
    [data-testid="stMain"] [data-baseweb="select"] * {
        color:#17233d !important;
    }
    [role="listbox"], [role="option"], [data-baseweb="popover"] {
        background-color:#ffffff !important;
        color:#17233d !important;
    }
    [role="option"] * {color:#17233d !important;}
    [role="option"]:hover {background:#e3eaff !important;}

    /* Inputs, uploader, buttons, and informational banners. */
    [data-testid="stFileUploaderDropzone"],
    [data-testid="stFileUploaderDropzone"] * {
        color:#17233d !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        background:#ffffff !important;
        border:1px dashed #8fa6d8 !important;
        border-radius:12px !important;
    }
    [data-testid="stFileUploaderDropzone"] button,
    [data-testid="stBaseButton-secondary"],
    [data-testid="stBaseButton-tertiary"] {
        background:#ffffff !important;
        color:#17233d !important;
        border-color:#aab9d7 !important;
    }
    [data-testid="stFileUploaderDropzone"] button *,
    [data-testid="stBaseButton-secondary"] *,
    [data-testid="stBaseButton-tertiary"] * {color:#17233d !important;}
    [data-testid="stBaseButton-primary"] {
        background:#4f46e5 !important;
        color:#ffffff !important;
    }
    [data-testid="stBaseButton-primary"] * {color:#ffffff !important;}
    [data-testid="stAlert"] {border-radius:12px !important;}
    [data-testid="stAlert"] p, [data-testid="stAlert"] span,
    [data-testid="stAlert"] [data-testid="stMarkdownContainer"] {
        color:#17233d !important;
    }
    [data-testid="stMain"] [data-testid="stDataFrame"] {
        background:#ffffff !important;
        color:#17233d !important;
    }
    [data-testid="stMain"] [data-testid="stSelectbox"] label,
    [data-testid="stMain"] [data-testid="stFileUploader"] label {
        color:#17233d !important;
    }
    [data-testid="stSidebar"] hr {border-color:#b9c6e4 !important;}
    @media (prefers-color-scheme: dark) {
        .stApp, [data-testid="stAppViewContainer"] {
            color-scheme:light !important;
            color:#17233d !important;
        }
        [data-testid="stSidebar"] {background:#eef4fb !important;}
    }


    /* FINAL CONTRAST RULES: preserve white text inside the dark hero banner.
       Streamlit's Markdown h1 theme styles otherwise override .hero h1. */
    html body .stApp .hero,
    html body .stApp .hero h1,
    html body .stApp .hero h1 *,
    html body .stApp .hero .hero-title {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        opacity: 1 !important;
    }
    html body .stApp .hero p,
    html body .stApp .hero p * {
        color: #eef6ff !important;
        -webkit-text-fill-color: #eef6ff !important;
    }
    html body .stApp .hero .eyebrow,
    html body .stApp .hero .eyebrow * {
        color: #c5e8ff !important;
        -webkit-text-fill-color: #c5e8ff !important;
    }
    html body .stApp .hero .hero-chip,
    html body .stApp .hero .hero-chip * {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    /* Light surfaces and dark readable text on sidebar, page, and controls. */
    html body .stApp [data-testid="stSidebar"] {
        background-color: #eef4fb !important;
        color: #1b2a3a !important;
    }
    html body .stApp [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    html body .stApp [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span,
    html body .stApp [data-testid="stSidebar"] [role="radiogroup"] label,
    html body .stApp [data-testid="stSidebar"] [role="radiogroup"] label * {
        color: #1b2a3a !important;
        -webkit-text-fill-color: #1b2a3a !important;
    }
    html body .stApp .section-title,
    html body .stApp .metric-card .value,
    html body .stApp .mini-stat .num {
        color: #102a46 !important;
        -webkit-text-fill-color: #102a46 !important;
    }
    html body .stApp .section-desc,
    html body .stApp .footer-note,
    html body .stApp .metric-card .hint {
        color: #506176 !important;
        -webkit-text-fill-color: #506176 !important;
    }
    html body .stApp [data-testid="stBaseButton-secondary"] {
        background-color: #ffffff !important;
        color: #1b2a3a !important;
        border-color: #c4d3e5 !important;
    }

</style>
""", unsafe_allow_html=True)

PALETTE = ["#4f46e5", "#0891b2", "#10b981", "#d97706", "#8b5cf6", "#2563eb"]

@st.cache_resource(show_spinner=False)
def load_model(path: str):
    return joblib.load(path)

@st.cache_data(show_spinner=False)
def load_metadata():
    with (ROOT / "metadata.json").open(encoding="utf-8") as f:
        return json.load(f)


def compute_results(model, X: pd.DataFrame, y: pd.Series):
    pred = model.predict(X)
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("Saved model does not support the required benign (1) class.")
    if hasattr(model, "predict_proba"):
        score = model.predict_proba(X)[:, classes.index(1)]
    elif hasattr(model, "decision_function"):
        raw = model.decision_function(X)
        score = raw if classes[-1] == 1 else -raw
    else:
        raise ValueError("This model cannot calculate ROC/AUC scores.")
    metrics = {
        "Accuracy": accuracy_score(y, pred),
        "AUC": roc_auc_score(y, score) if y.nunique() == 2 else None,
        "Precision": precision_score(y, pred, pos_label=1, zero_division=0),
        "Recall": recall_score(y, pred, pos_label=1, zero_division=0),
        "F1": f1_score(y, pred, pos_label=1, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }
    return {"pred": pred, "score": score, "metrics": metrics}


def metric_card(title, value, note, color):
    formatted = f"{value:.4f}" if value is not None else "N/A"
    st.markdown(f"""
    <div class="metric-card" style="--accent:{color}">
      <div class="label">{title}</div>
      <div class="value">{formatted}</div>
      <div class="hint">{note}</div>
    </div>""", unsafe_allow_html=True)


def mini_stat(label, value):
    st.markdown(f"<div class='mini-stat'><div class='lbl'>{label}</div>"
                f"<div class='num'>{value}</div></div>", unsafe_allow_html=True)


metadata = load_metadata()
features = metadata["features"]
target_col = metadata["target"]
model_names = list(metadata["model_files"])

st.markdown("""
<div class="hero">
    <div class="eyebrow">Machine Learning · Evaluation Studio</div>
    <h1>Breast Tumor Classification Lab</h1>
    <p>Explore five trained classifiers, inspect prediction quality and compare performance
       on the <b>same uploaded test dataset</b>. Models are never retrained here.</p>
    <span class="hero-chip">🧬 30 features</span>
    <span class="hero-chip">📊 6 evaluation metrics</span>
    <span class="hero-chip">🔬 5 classifiers</span>
    <span class="hero-chip">🎓 Educational only</span>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🧬 Model Lab")
    st.caption("Explore the diagnostic classification experiment.")
    st.divider()
    section = st.radio(
        "NAVIGATION",
        ["📂 Upload Data", "🎯 Individual Model", "📊 Compare All 5 Models", "📈 ROC Curves"],
        key="active_section",
        label_visibility="visible",
    )
    st.divider()
    selected_model = st.selectbox("Focus model", model_names, index=0)
    if st.session_state.get("test_csv_bytes"):
        st.success("✓ Test dataset ready")
        if st.button("Clear uploaded dataset", use_container_width=True):
            st.session_state.pop("test_csv_bytes", None)
            st.session_state.pop("test_csv_name", None)
            st.session_state.pop("upload_widget", None)
            st.rerun()
    else:
        st.info("No test data uploaded")
    st.divider()
    st.markdown("**Class labels**")
    st.markdown("🔴 **0 — Malignant**")
    st.markdown("🟢 **1 — Benign**")
    st.caption("AUC, precision, recall, F1 use benign (1) as the positive class.")
    st.caption("Educational use only — not for clinical decisions.")

if section == "📂 Upload Data":
    st.markdown("<div class='section-title'>Upload test dataset</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-desc'>Upload your held-out Wisconsin Diagnostic CSV here. The file is used across all three evaluation views.</div>", unsafe_allow_html=True)
    left, right = st.columns([1.5, 1])
    with left:
        uploaded_file = st.file_uploader(
            "Choose a CSV file containing all 30 features and target",
            type=["csv"], accept_multiple_files=False, key="upload_widget"
        )
        if uploaded_file is not None:
            new_bytes = uploaded_file.getvalue()
            if (st.session_state.get("test_csv_bytes") != new_bytes
                    or st.session_state.get("test_csv_name") != uploaded_file.name):
                st.session_state["test_csv_bytes"] = new_bytes
                st.session_state["test_csv_name"] = uploaded_file.name
        if st.session_state.get("test_csv_bytes"):
            st.success(f"Loaded: {st.session_state['test_csv_name']}")
        else:
            st.info("Upload the test CSV to unlock evaluation views.")
    with right:
        st.markdown("#### Required data format")
        st.write("30 numerical feature columns and one `target` column.")
        st.write("Class labels: **0 = malignant**, **1 = benign**")
        sample_path = ROOT / "test_data.csv"
        if sample_path.is_file():
            st.download_button("⬇ Download example test CSV", sample_path.read_bytes(),
                               file_name="test_data.csv", mime="text/csv", use_container_width=True)

if not st.session_state.get("test_csv_bytes"):
    if section != "📂 Upload Data":
        st.warning("No dataset uploaded. Select **Upload Data** in the left sidebar to begin.")
        if st.button("Go to Upload Data"):
            st.session_state.active_section = "📂 Upload Data"
            st.rerun()
    st.stop()

from io import BytesIO
uploaded = BytesIO(st.session_state["test_csv_bytes"])

try:
    uploaded.seek(0)
    df = pd.read_csv(uploaded)
except Exception as exc:
    st.error(f"Could not read the uploaded CSV: {exc}")
    st.stop()

required = features + [target_col]
missing = [col for col in required if col not in df.columns]
if missing:
    st.error("Invalid dataset: the uploaded CSV does not contain the expected Wisconsin Diagnostic feature structure.")
    st.warning(f"Missing {len(missing)} required columns: " + ", ".join(missing))
    st.stop()
if df.empty:
    st.error("The uploaded CSV contains no rows.")
    st.stop()
try:
    X = df[features].apply(pd.to_numeric, errors="raise")
    y_raw = pd.to_numeric(df[target_col], errors="raise")
except (TypeError, ValueError):
    st.error("Input features and target labels must all be numeric.")
    st.stop()
if X.isna().any().any() or y_raw.isna().any():
    st.error("Missing feature or target values are not supported.")
    st.stop()
if not y_raw.isin([0, 1]).all():
    st.error("Invalid target labels. Only 0 (malignant) and 1 (benign) are supported.")
    st.stop()
if (X < 0).any().any():
    st.error("Tumor measurement features must not contain negative values.")
    st.stop()
y = y_raw.astype(int)
extra = [c for c in df.columns if c not in required]
if extra:
    st.caption(f"Ignoring {len(extra)} additional column(s); predictions use only the saved model's 30 features.")

# Evaluate all five saved models against the SAME uploaded test rows.
results = {}
try:
    with st.spinner("Evaluating the saved models on your uploaded CSV…"):
        for name in model_names:
            model_path = ROOT / metadata["model_files"][name]
            if not model_path.is_file():
                raise FileNotFoundError(f"Saved model missing: {model_path.name}")
            model = load_model(str(model_path))
            results[name] = compute_results(model, X, y)
except Exception as exc:
    st.error(f"Evaluation could not complete: {exc}")
    st.stop()

if section == "📂 Upload Data":
    st.markdown("<div class='section-title'>Validation results</div>", unsafe_allow_html=True)
    st.markdown("<div class='eyebrow-dark'>Uploaded dataset</div>", unsafe_allow_html=True)
    statcols = st.columns(4)
    for col, title, val in zip(statcols,
            ["Test records", "Input features", "Malignant (0)", "Benign (1)"],
            [len(y), len(features), int((y == 0).sum()), int((y == 1).sum())]):
        with col:
            mini_stat(title, val)

    st.markdown("<div class='data-note'>All five models use the same uploaded test data. No fitting or training is performed during evaluation.</div>", unsafe_allow_html=True)

    st.success("Dataset validated successfully. Use the left sidebar to open an evaluation view.")
    st.dataframe(df.head(8), use_container_width=True, hide_index=True)
    st.stop()

if section == "🎯 Individual Model":
    st.markdown(f"<div class='section-title'>{selected_model} · Test results</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-desc'>Scores from the selected pre-trained model. Higher values are better for all six metrics.</div>", unsafe_allow_html=True)
    selected = results[selected_model]
    metric_cols = st.columns(3)
    for i, (key, val) in enumerate(selected["metrics"].items()):
        with metric_cols[i % 3]:
            hint = "Not defined for a single-class dataset" if val is None else (
                "Benign (1) is the positive class" if key in {"Precision", "Recall", "F1", "AUC"}
                else "Computed from uploaded test rows")
            metric_card("F1 Score" if key == "F1" else key, val, hint, PALETTE[i])

    cm = confusion_matrix(y, selected["pred"], labels=[0, 1])
    matrix_col, details_col = st.columns([1.08, 1])
    with matrix_col:
        st.markdown("#### Confusion matrix")
        fig, ax = plt.subplots(figsize=(6.0, 4.2), dpi=130)
        cmap = plt.get_cmap("Blues")
        ax.imshow(cm, cmap=cmap, vmin=0, vmax=max(int(cm.max()), 1))
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=18,
                        fontweight="bold", color="white" if cm[i, j] > cm.max() / 2 else "#183153")
        ax.set_xticks([0, 1], ["Malignant (0)", "Benign (1)"])
        ax.set_yticks([0, 1], ["Malignant (0)", "Benign (1)"])
        ax.set_xlabel("Predicted label", labelpad=12)
        ax.set_ylabel("Actual label", labelpad=12)
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.patch.set_facecolor("white")
        fig.tight_layout()
        fig.subplots_adjust(left=0.24, bottom=0.20)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with details_col:
        st.markdown("#### Prediction breakdown")
        good = int(np.trace(cm))
        bad = int(len(y) - good)
        st.success(f"**{good} correctly classified** out of {len(y)} records")
        st.error(f"**{bad} misclassified** records")
        breakdown = pd.DataFrame([
            {"Outcome": "Malignant → malignant", "Records": int(cm[0, 0]), "Interpretation": "Correct"},
            {"Outcome": "Malignant → benign", "Records": int(cm[0, 1]), "Interpretation": "Incorrect"},
            {"Outcome": "Benign → malignant", "Records": int(cm[1, 0]), "Interpretation": "Incorrect"},
            {"Outcome": "Benign → benign", "Records": int(cm[1, 1]), "Interpretation": "Correct"},
        ])
        st.dataframe(breakdown, use_container_width=True, hide_index=True)
        st.caption("Rows represent actual labels; columns represent predictions. Correct results lie on the diagonal.")

    st.markdown("#### Classification report")
    report_dict = classification_report(
        y, selected["pred"], labels=[0, 1], target_names=["Malignant (0)", "Benign (1)"],
        output_dict=True, zero_division=0,
    )
    report_rows = []
    for name in ["Malignant (0)", "Benign (1)", "macro avg", "weighted avg"]:
        item = report_dict[name]
        report_rows.append({
            "Class / Average": name,
            "Precision": item["precision"],
            "Recall": item["recall"],
            "F1 Score": item["f1-score"],
            "Support": int(item["support"]),
        })
    report_df = pd.DataFrame(report_rows)
    st.dataframe(report_df.style.format({"Precision": "{:.4f}", "Recall": "{:.4f}",
                                         "F1 Score": "{:.4f}", "Support": "{:d}"}),
                 use_container_width=True, hide_index=True)
    st.caption(f"Overall accuracy: {selected['metrics']['Accuracy']:.4f} ({good} correct of {len(y)}).")

    prediction_csv = df.copy()
    prediction_csv["predicted_target"] = selected["pred"]
    prediction_csv["predicted_label"] = np.where(selected["pred"] == 1, "benign", "malignant")
    prediction_csv["benign_probability"] = selected["score"]
    prediction_csv["is_correct"] = (selected["pred"] == y.to_numpy())
    st.download_button("⬇ Download selected-model predictions",
                       data=prediction_csv.to_csv(index=False).encode("utf-8"),
                       file_name=f"{selected_model.lower().replace(' ', '_')}_predictions.csv",
                       mime="text/csv")

if section == "📊 Compare All 5 Models":
    st.markdown("<div class='section-title'>Compare all five classifiers</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-desc'>Each model is evaluated on the identical uploaded test dataset. Values are computed live from saved models.</div>", unsafe_allow_html=True)
    compare_rows = [{"Model": name, **results[name]["metrics"]} for name in model_names]
    compare_df = pd.DataFrame(compare_rows)
    formatted = compare_df.style.format({col: "{:.4f}" for col in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]}, na_rep="N/A")
    st.dataframe(formatted, use_container_width=True, hide_index=True)
    if y.nunique() == 2:
        max_acc = compare_df["Accuracy"].max()
        tied = compare_df[np.isclose(compare_df["Accuracy"], max_acc)]
        # Rank highest accuracy first, then AUC for meaningful tie-breaking.
        winner = tied.sort_values("AUC", ascending=False).iloc[0]
        st.success(f"🏆 Highest accuracy: **{winner['Model']}** — {winner['Accuracy']:.4f} accuracy, "
                   f"{winner['AUC']:.4f} AUC (AUC breaks accuracy ties).")
    else:
        st.warning("Uploaded data contains one class: AUC cannot be calculated or used to break ties.")
    metric_choice = st.selectbox("Metric to visualize", ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"], index=0)
    chart_data = compare_df.set_index("Model")[[metric_choice]].dropna()
    if len(chart_data):
        fig, ax = plt.subplots(figsize=(9, 3.8), dpi=125)
        labels = chart_data.index.tolist()[::-1]
        values = chart_data[metric_choice].tolist()[::-1]
        bars = ax.barh(labels, values, color="#4f46e5", height=.58)
        for bar, value in zip(bars, values):
            ax.text(bar.get_width() + .012, bar.get_y() + bar.get_height()/2,
                    f"{value:.4f}", va="center", fontsize=9, color="#263553")
        xmin = min(0.0, min(values) - .1) if values else 0
        ax.set_xlim(xmin, max(1.10, max(values) + .12))
        ax.set_xlabel(metric_choice)
        ax.grid(axis="x", alpha=.13)
        ax.set_axisbelow(True)
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.info("This metric is not defined for the uploaded data.")
    st.download_button("⬇ Download model comparison CSV", compare_df.to_csv(index=False).encode("utf-8"),
                       file_name="model_comparison.csv", mime="text/csv")

if section == "📈 ROC Curves":
    st.markdown("<div class='section-title'>ROC curve comparison</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-desc'>Receiver operating characteristic curves show the trade-off between true-positive and false-positive rates.</div>", unsafe_allow_html=True)
    if y.nunique() != 2:
        st.warning("ROC curves require both malignant (0) and benign (1) labels in the uploaded test CSV.")
    else:
        fig, ax = plt.subplots(figsize=(9.0, 5.5), dpi=125)
        for i, name in enumerate(model_names):
            fpr, tpr, _ = roc_curve(y, results[name]["score"], pos_label=1)
            auc = results[name]["metrics"]["AUC"]
            ax.plot(fpr, tpr, lw=3.2 if name == selected_model else 2,
                    alpha=1.0 if name == selected_model else .78,
                    color=PALETTE[i], label=f"{name} (AUC {auc:.4f})")
        ax.plot([0, 1], [0, 1], "--", color="#94a3b8", lw=1.5, label="Chance (AUC 0.5000)")
        ax.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.02))
        ax.grid(alpha=.15)
        ax.set_facecolor("#fbfdff")
        ax.legend(loc="lower right", framealpha=.95, fontsize=9)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption("Positive label = benign (1). Highlighted curve corresponds to the selected focus model.")

st.markdown("<div class='footer-note'>This educational app compares pre-trained classifiers only. "
            "Structural CSV validation does not guarantee data provenance or clinical suitability. "
            "For reliable evaluation, use the untouched held-out test data. "
            "Predictions must not be used for medical decisions.</div>", unsafe_allow_html=True)
