
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from imblearn.over_sampling import SMOTE

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Telco Customer Churn Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# DESIGN SYSTEM / CUSTOM CSS
# =========================================================
st.markdown("""
<style>
    [data-testid="stSidebarNav"] {
        display: none;
    }

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: #F7F9FC;
    }

    [data-testid="stHeader"] {
        background: rgba(247, 249, 252, 0.85);
    }

    /* =====================================================
       SIDEBAR - NAVY BACKGROUND + WHITE FILTER BOXES
       ===================================================== */

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #182538 0%, #21324A 100%) !important;
        border-right: 1px solid rgba(255,255,255,0.08) !important;
    }

    [data-testid="stSidebar"] > div {
        background: linear-gradient(180deg, #1D2E45 0%, #263A55 100%) !important;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stMarkdown * {
        color: #F8FAFC !important;
        opacity: 1 !important;
    }

    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.16) !important;
    }

    /* Sidebar buttons */
    [data-testid="stSidebar"] .stButton > button {
        background-color: #304761 !important;
        color: #23324A !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 14px !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.10) !important;
        transition: all 0.2s ease !important;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        background-color: #49627E !important;
        border-color: #B7CBE1 !important;
        color: #23324A !important;
    }

    [data-testid="stSidebar"] [data-testid="stSlider"] * {
        color: #F8FAFC !important;
    }

    .block-container {
        padding-top: 2.5rem;
        padding-bottom: 3rem;
        max-width: 1240px;
    }

    .hero-wrap {
        background: linear-gradient(135deg, #FFFFFF 0%, #EEF6FF 100%);
        border: 1px solid #E4E7EC;
        border-radius: 24px;
        padding: 30px 32px;
        box-shadow: 0 12px 28px rgba(16, 24, 40, 0.06);
        margin-bottom: 22px;
    }

    .hero-title {
        font-size: 48px;
        line-height: 1.05;
        font-weight: 800;
        letter-spacing: -1.5px;
        color: #23324A;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        color: #5F6B7A;
        font-size: 16.5px;
        line-height: 1.7;
        max-width: 980px;
        margin-bottom: 0px;
    }

    .section-title {
        font-size: 30px;
        font-weight: 800;
        letter-spacing: -0.8px;
        color: #23324A;
        margin-top: 38px;
        margin-bottom: 6px;
    }

    .section-caption {
        color: #697586;
        font-size: 15.5px;
        margin-bottom: 20px;
    }

    .metric-card {
        border: 1px solid #E4E7EC;
        border-radius: 18px;
        background: #FFFFFF;
        padding: 22px 20px 16px 20px;
        box-shadow: 0 8px 24px rgba(16, 24, 40, 0.06);
        min-height: 156px;
        transition: all 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 28px rgba(16, 24, 40, 0.10);
    }

    .metric-label {
        color: #667085;
        font-size: 14px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .metric-value {
        color: #23324A;
        font-size: 36px;
        font-weight: 800;
        letter-spacing: -1px;
        margin-bottom: 8px;
    }

    .metric-up {
        background-color: #DCFCE7;
        color: #16A34A;
        border-radius: 999px;
        padding: 5px 11px;
        font-size: 13px;
        font-weight: 800;
    }

    .metric-down {
        background-color: #FEE2E2;
        color: #DC2626;
        border-radius: 999px;
        padding: 5px 11px;
        font-size: 13px;
        font-weight: 800;
    }

    .info-box {
        background: #EEF6FF;
        color: #1D4ED8;
        border-radius: 16px;
        padding: 18px 20px;
        font-weight: 650;
        border: 1px solid #D8EAFF;
        line-height: 1.55;
    }

    .soft-card {
        border: 1px solid #E4E7EC;
        border-radius: 18px;
        background: #FFFFFF;
        padding: 18px;
        box-shadow: 0 8px 24px rgba(16, 24, 40, 0.05);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #EEF2F7;
        border-radius: 999px;
        padding: 10px 18px;
        color: #344054;
        font-weight: 700;
    }

    .stTabs [aria-selected="true"] {
        background-color: #23324A;
        color: white;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 16px;
        overflow: hidden;
    }

    hr {
        margin: 1.8rem 0;
        border-color: #E4E7EC;
    }

    .group-card {
        margin-top: 22px;
        padding: 18px 20px;
        background: #ffffff;
        border: 1px solid #e4e7ec;
        border-radius: 16px;
        color: #23324A;
        font-size: 16px;
        line-height: 1.8;
        box-shadow: 0 8px 24px rgba(16, 24, 40, 0.06);
        max-width: 720px;
    }

    /* =====================================================
       CHART EXPLANATION CARD
       ===================================================== */
    .chart-insight {
        background: #FFFFFF;
        border: 1px solid #D8EAFF;
        border-left: 6px solid #3B73B9;
        border-radius: 16px;
        padding: 16px 18px;
        margin-top: -6px;
        margin-bottom: 24px;
        box-shadow: 0 8px 22px rgba(16, 24, 40, 0.05);
        color: #23324A;
        font-size: 14px;
        line-height: 1.65;
    }

    .chart-insight-title {
        font-size: 14px;
        font-weight: 800;
        color: #23324A;
        margin-bottom: 6px;
    }

    .chart-insight-text {
        color: #5F6B7A;
        font-weight: 500;
        margin-bottom: 10px;
    }

    .chart-conclusion {
        background: #EEF6FF;
        border: 1px solid #D8EAFF;
        border-radius: 12px;
        padding: 10px 12px;
        color: #23324A;
        font-weight: 600;
    }

    /* =====================================================
   FINAL SIDEBAR FILTER STYLE
   ===================================================== */

/* Kotak filter tetap navy */
[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background-color: #23324A !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 16px !important;
    color: #FFFFFF !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08) !important;
}

/* Tulisan untuk selectbox biasa seperti All */
[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] * {
    color: #FFFFFF !important;
}

/* Arrow selectbox biasa */
[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] svg {
    fill: #FFFFFF !important;
    color: #FFFFFF !important;
    opacity: 1 !important;
}

/* Multiselect container tetap navy */
[data-testid="stSidebar"] [data-testid="stMultiSelect"] div[data-baseweb="select"] > div {
    background-color: #23324A !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 16px !important;
}

/* Tag pilihan multiselect putih */
[data-testid="stSidebar"] [data-testid="stMultiSelect"] [data-baseweb="tag"] {
    background: #FFFFFF !important;
    background-color: #FFFFFF !important;
    color: #23324A !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 10px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.12) !important;
}

/* Tulisan dalam tag multiselect: Month-to-month, One year, DSL, dll */
[data-testid="stSidebar"] [data-testid="stMultiSelect"] [data-baseweb="tag"] *,
[data-testid="stSidebar"] [data-testid="stMultiSelect"] [data-baseweb="tag"] span,
[data-testid="stSidebar"] [data-testid="stMultiSelect"] [data-baseweb="tag"] div {
    color: #23324A !important;
    background: transparent !important;
    background-color: transparent !important;
    opacity: 1 !important;
    visibility: visible !important;
}

/* X kecil di dalam tag tetap navy */
[data-testid="stSidebar"] [data-testid="stMultiSelect"] [data-baseweb="tag"] svg {
    fill: #23324A !important;
    color: #23324A !important;
    opacity: 1 !important;
}

/* X clear kanan dan arrow dropdown di multiselect tetap putih */
[data-testid="stSidebar"] [data-testid="stMultiSelect"] div[data-baseweb="select"] svg {
    fill: #FFFFFF !important;
    color: #FFFFFF !important;
    opacity: 1 !important;
}

/* Hide typing/search field tanpa bikin value hilang */
[data-testid="stSidebar"] div[data-baseweb="select"] input {
    width: 1px !important;
    min-width: 1px !important;
    max-width: 1px !important;
    padding: 0 !important;
    margin: 0 !important;
    opacity: 0 !important;
    caret-color: transparent !important;
}
            
    /* Force white arrow - semua kemungkinan selector */
[data-testid="stSidebarCollapseButton"] button,
[data-testid="stSidebarCollapseButton"] button span,
[data-testid="stSidebarCollapseButton"] svg,
[data-testid="stSidebarCollapseButton"] svg path,
[data-testid="stSidebarCollapseButton"] * {
    color: #FFFFFF !important;
}

[data-testid="stSidebarCollapseButton"] button {
    background: transparent !important;
    filter: brightness(10) !important;  /* nuclear option */
}        

</style>
""", unsafe_allow_html=True)

# =========================================================
# DATA & NOTEBOOK SETTINGS
# =========================================================
DATA_URL = "https://raw.githubusercontent.com/HEHEfebrian/ALPDS/refs/heads/main/WA_Fn-UseC_-Telco-Customer-Churn.csv"

PREDICTION_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "PaperlessBilling",
    "MonthlyCharges",
    "Contract_One year",
    "Contract_Two year",
    "PaymentMethod_Credit card (automatic)",
    "PaymentMethod_Electronic check",
    "PaymentMethod_Mailed check",
]

DROP_MULTICOL_COLUMNS = [
    "MultipleLines_No phone service",
    "InternetService_No",
    "OnlineSecurity_No internet service",
    "OnlineBackup_No internet service",
    "DeviceProtection_No internet service",
    "TechSupport_No internet service",
    "StreamingTV_No internet service",
    "StreamingMovies_No internet service",
    "TotalCharges",
    "InternetService_Fiber optic",
    "MultipleLines_Yes",
    "OnlineSecurity_Yes",
    "OnlineBackup_Yes",
    "DeviceProtection_Yes",
    "TechSupport_Yes",
    "StreamingTV_Yes",
    "StreamingMovies_Yes",
]

BEST_PARAMS = {
    "Logistic Regression": {
        "non_smote": {"C": 0.01, "class_weight": "balanced", "max_iter": 1000, "solver": "liblinear"},
        "smote": {"C": 10, "class_weight": None, "max_iter": 1000, "solver": "liblinear"},
    },
    "K-Nearest Neighbors": {
        "non_smote": {"metric": "manhattan", "n_neighbors": 21, "weights": "uniform"},
        "smote": {"metric": "manhattan", "n_neighbors": 15, "weights": "distance"},
    },
    "Naive Bayes": {
        "non_smote": {"var_smoothing": 0.12328467394420659},
        "smote": {"var_smoothing": 0.008111308307896872},
    },
    "Random Forest": {
        "non_smote": {
            "class_weight": "balanced",
            "max_depth": 5,
            "max_features": "sqrt",
            "min_samples_leaf": 2,
            "min_samples_split": 2,
            "n_estimators": 200,
            "random_state": 42,
        },
        "smote": {
            "class_weight": "balanced",
            "max_depth": 20,
            "max_features": "sqrt",
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "n_estimators": 200,
            "random_state": 42,
        },
    },
}

# =========================================================
# DATA FUNCTIONS
# =========================================================
@st.cache_data(show_spinner=False)
def load_data():
    raw_df = pd.read_csv(DATA_URL)
    raw_df = raw_df.replace(r"^\s*$", np.nan, regex=True)
    raw_df["TotalCharges"] = pd.to_numeric(raw_df["TotalCharges"], errors="coerce")
    raw_df["TotalCharges"] = raw_df["TotalCharges"].fillna(raw_df["MonthlyCharges"] * raw_df["tenure"])
    raw_df = raw_df.drop_duplicates()
    return raw_df


def preprocess_for_model(raw_df):
    model_df = raw_df.copy()

    scaler = StandardScaler()
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    model_df[num_cols] = scaler.fit_transform(model_df[num_cols])

    binary_cols = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]
    for col in binary_cols:
        model_df[col] = model_df[col].map({"Yes": 1, "No": 0, "Male": 1, "Female": 0})

    model_df = pd.get_dummies(
        model_df,
        columns=[
            "InternetService",
            "Contract",
            "PaymentMethod",
            "MultipleLines",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ],
        drop_first=True,
    )

    model_df = model_df.drop("customerID", axis=1)
    bool_cols = model_df.select_dtypes(include="bool").columns
    model_df[bool_cols] = model_df[bool_cols].astype(int)
    model_df["Churn"] = model_df.pop("Churn")
    return model_df, scaler


@st.cache_resource(show_spinner=False)
def train_models(raw_df):
    model_df, scaler = preprocess_for_model(raw_df)

    X = model_df.drop("Churn", axis=1)
    y = model_df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    existing_drop_cols = [c for c in DROP_MULTICOL_COLUMNS if c in X_train.columns]
    X_train = X_train.drop(columns=existing_drop_cols)
    X_test = X_test.drop(columns=existing_drop_cols)

    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    def make_model(model_name, mode):
        params = BEST_PARAMS[model_name][mode]
        if model_name == "Logistic Regression":
            return LogisticRegression(**params)
        if model_name == "K-Nearest Neighbors":
            return KNeighborsClassifier(**params)
        if model_name == "Naive Bayes":
            return GaussianNB(**params)
        if model_name == "Random Forest":
            return RandomForestClassifier(**params)
        raise ValueError(model_name)

    results = {}
    for model_name in BEST_PARAMS.keys():
        for mode, train_x, train_y in [
            ("non_smote", X_train, y_train),
            ("smote", X_train_smote, y_train_smote),
        ]:
            model = make_model(model_name, mode)
            model.fit(train_x, train_y)
            pred = model.predict(X_test)

            label = model_name if mode == "non_smote" else f"{model_name} (SMOTE)"
            results[label] = {
                "model": model,
                "base_model": model_name,
                "mode": "Non-SMOTE" if mode == "non_smote" else "SMOTE",
                "accuracy": accuracy_score(y_test, pred),
                "precision": precision_score(y_test, pred, zero_division=0),
                "recall": recall_score(y_test, pred, zero_division=0),
                "f1": f1_score(y_test, pred, zero_division=0),
                "cm": confusion_matrix(y_test, pred),
                "report": classification_report(y_test, pred, output_dict=True, zero_division=0),
                "best_params": BEST_PARAMS[model_name][mode],
            }

    model_info = {
        "scaler": scaler,
        "prediction_columns": X_train.columns.tolist(),
        "before_smote": y_train.value_counts().to_dict(),
        "after_smote": y_train_smote.value_counts().to_dict(),
        "test_distribution": y_test.value_counts().to_dict(),
        "dropped_columns": existing_drop_cols,
    }

    return results, model_info


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def metric_card(label, value, delta, positive=True):
    cls = "metric-up" if positive else "metric-down"
    arrow = "↑" if positive else "↓"
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <span class="{cls}">{arrow} {delta}</span>
        </div>
        """,
        unsafe_allow_html=True
    )


def chart_explanation(explanation, conclusion):
    st.markdown(
        f"""
        <div class="chart-insight">
            <div class="chart-insight-title">💡 Explanation</div>
            <div class="chart-insight-text">{explanation}</div>
            <div class="chart-conclusion"><b>Conclusion:</b> {conclusion}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def clean_plotly(fig, height=380):
    fig.update_layout(
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=20, r=20, t=55, b=35),
        font=dict(family="Inter", color="#23324A", size=13),
        title_font=dict(size=18, color="#23324A"),
        legend=dict(orientation="h", y=-0.20, x=0.5, xanchor="center", font=dict(color="#23324A")),
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="#E7DDCF",
        zeroline=False,
        title_font=dict(color="#23324A"),
        tickfont=dict(color="#23324A")
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="#E7DDCF",
        zeroline=False,
        title_font=dict(color="#23324A"),
        tickfont=dict(color="#23324A")
    )

    for trace_type in ["bar", "pie", "scatter", "heatmap"]:
        fig.update_traces(
            selector=dict(type=trace_type),
            textfont=dict(color="#23324A", size=13)
        )

    fig.update_layout(
        coloraxis_colorbar=dict(
            title=dict(font=dict(color="#23324A", size=14)),
            tickfont=dict(color="#23324A", size=13)
        )
    )
    return fig


def make_churn_rate_table(data, group_col):
    table = data.groupby(group_col, as_index=False).agg(
        Customers=("customerID", "count"),
        Churners=("Churn", lambda x: (x == "Yes").sum())
    )
    table["Churn Rate"] = np.where(table["Customers"] > 0, table["Churners"] / table["Customers"] * 100, 0)
    return table


def metrics_dataframe(results):
    return pd.DataFrame([
        {
            "Model": name,
            "Type": info["mode"],
            "Accuracy": info["accuracy"],
            "Precision": info["precision"],
            "Recall": info["recall"],
            "F1-Score": info["f1"],
        }
        for name, info in results.items()
    ])


# =========================================================
# LOAD APP DATA
# =========================================================
df = load_data()
models, model_info = train_models(df)

# =========================================================
# SIDEBAR FILTERS
# =========================================================
with st.sidebar:
    st.markdown("## 📡 Menu")

    st.button("📊 Dashboard", use_container_width=True, disabled=True)

    if st.button("👥 Member", use_container_width=True):
        st.switch_page("pages/member.py")

    st.markdown("---")
    st.markdown("## ⚗ Filters")

    churn_filter = st.selectbox("Churn Status", ["All", "No", "Yes"])

    contract_filter = st.multiselect(
        "Contract",
        options=sorted(df["Contract"].unique()),
        default=sorted(df["Contract"].unique())
    )

    internet_filter = st.multiselect(
        "Internet Service",
        options=sorted(df["InternetService"].unique()),
        default=sorted(df["InternetService"].unique())
    )

    payment_filter = st.multiselect(
        "Payment Method",
        options=sorted(df["PaymentMethod"].unique()),
        default=sorted(df["PaymentMethod"].unique())
    )

    senior_filter = st.selectbox("Senior Citizen", options=["All", "No", "Yes"])

    tenure_range = st.slider(
        "Tenure Range (Months)",
        int(df["tenure"].min()),
        int(df["tenure"].max()),
        (int(df["tenure"].min()), int(df["tenure"].max()))
    )


# Safety fallback: if a user clears all options using the X button,
# keep the dashboard and navigation usable by returning to "All" values.
if not contract_filter:
    contract_filter = sorted(df["Contract"].unique())

if not internet_filter:
    internet_filter = sorted(df["InternetService"].unique())

if not payment_filter:
    payment_filter = sorted(df["PaymentMethod"].unique())


filtered_df = df[
    (df["Contract"].isin(contract_filter)) &
    (df["InternetService"].isin(internet_filter)) &
    (df["PaymentMethod"].isin(payment_filter)) &
    (df["tenure"].between(tenure_range[0], tenure_range[1]))
]

if churn_filter != "All":
    filtered_df = filtered_df[filtered_df["Churn"] == churn_filter]

if senior_filter != "All":
    senior_value = 1 if senior_filter == "Yes" else 0
    filtered_df = filtered_df[filtered_df["SeniorCitizen"] == senior_value]

# =========================================================
# HERO
# =========================================================
st.markdown(
"""
<div class="hero-wrap">
    <div class="hero-title">📡 Telco Customer Churn Dashboard</div>
    <div class="hero-subtitle">
        This dashboard shows customer churn analysis in the telecommunication industry.
        It helps us understand customer behavior, find factors that may cause churn,
        and compare machine learning models to predict whether a customer will churn or not.
        <br><br>
        The analysis follows the latest <b>ALP CEK notebook</b>, including data cleaning,
        standardization, encoding, multicollinearity removal, SMOTE, model training,
        and model evaluation.
    </div>
</div>
""",
unsafe_allow_html=True
)

notebook_path = "/Users/caseydaniellawinarto/KERJA DISINI YAH KES/streamlit-datascience/CEK_Python Notebook for ALP.ipynb"

col_kaggle, col_nb, col_empty = st.columns([7,6,7])

with col_kaggle:
    st.link_button(
        "🔗 Open Telco Customer Churn Dataset on Kaggle",
        "https://www.kaggle.com/datasets/blastchar/telco-customer-churn"
    )

from pathlib import Path
import streamlit as st

notebook_path = Path("CEK_Python Notebook for ALP.ipynb")

if notebook_path.exists():
    with open(notebook_path, "rb") as f:
        st.download_button(
            label="📓 Download ALP CEK Notebook (.ipynb)",
            data=f.read(),
            file_name="CEK_Python Notebook for ALP.ipynb",
            mime="application/octet-stream"
        )
else:
    st.error(f"Notebook tidak ditemukan: {notebook_path}")

# =========================================================
# KPI
# =========================================================
total_customers = len(filtered_df)
churn_count = (filtered_df["Churn"] == "Yes").sum()
not_churn_count = (filtered_df["Churn"] == "No").sum()
churn_rate = churn_count / total_customers * 100 if total_customers else 0
avg_monthly = filtered_df["MonthlyCharges"].mean() if total_customers else 0
avg_tenure = filtered_df["tenure"].mean() if total_customers else 0

k1, k2, k3, k4 = st.columns(4)
with k1:
    metric_card("Total Customers", f"{total_customers:,}", "Filtered data", True)
with k2:
    metric_card("Churn Rate", f"{churn_rate:.1f}%", f"{churn_count:,} churners", churn_rate < 30)
with k3:
    metric_card("Avg. Monthly Charges", f"${avg_monthly:.2f}", "Monthly", True)
with k4:
    metric_card("Avg. Tenure", f"{avg_tenure:.1f}", "Months", True)

# =========================================================
# PRELIMINARY ANALYSIS
# =========================================================
st.markdown('<div class="section-title">① Preliminary Analysis</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-caption">Initial data overview following the notebook: churn distribution, internet service distribution, and numerical feature distribution.</div>',
    unsafe_allow_html=True
)

p1, p2, p3 = st.columns([1, 1, 1.2])

with p1:
    churn_dist = filtered_df["Churn"].value_counts().reset_index()
    churn_dist.columns = ["Churn", "Count"]
    fig = px.bar(
        churn_dist,
        x="Churn",
        y="Count",
        text="Count",
        color="Churn",
        title="Customer Churn Distribution",
        color_discrete_map={"Yes": "#FF6B6B", "No": "#1971C2"}
    )
    st.plotly_chart(clean_plotly(fig, 360), use_container_width=True)

    chart_explanation(
        "This bar chart shows how many customers did not churn and how many customers churned.",
        "Most customers did not churn, but the churn group is still important because it represents customers who stopped using the service."
    )

with p2:
    internet_dist = filtered_df["InternetService"].value_counts().reset_index()
    internet_dist.columns = ["Internet Service", "Count"]
    fig = px.pie(
        internet_dist,
        names="Internet Service",
        values="Count",
        hole=0.45,
        title="Internet Service Distribution",
    )
    st.plotly_chart(clean_plotly(fig, 360), use_container_width=True)

    chart_explanation(
        "This donut chart shows the proportion of customers by internet service type: Fiber optic, DSL, and no internet service.",
        "Internet service type is useful for understanding customer behavior because different service types can have different churn patterns."
    )

with p3:
    selected_numeric = st.selectbox(
        "Choose numerical feature",
        ["tenure", "MonthlyCharges", "TotalCharges"],
        index=1
    )
    fig = px.histogram(
        filtered_df,
        x=selected_numeric,
        nbins=30,
        color="Churn",
        marginal="box",
        title=f"Distribution of {selected_numeric}",
        color_discrete_map={"Yes": "#FF6B6B", "No": "#1971C2"}
    )
    st.plotly_chart(clean_plotly(fig, 360), use_container_width=True)

    chart_explanation(
        "This chart shows the distribution of the selected numerical feature and compares churn versus non-churn customers.",
        "The distribution helps us see whether features like MonthlyCharges, tenure, or TotalCharges show different patterns for churn customers."
    )

# =========================================================
# RELATIONSHIP & CORRELATION
# =========================================================
st.markdown('<div class="section-title">② Relationship & Correlation</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-caption">Relationship between numerical features and churn status using correlation heatmap and boxplot.</div>',
    unsafe_allow_html=True
)

r1, r2 = st.columns([1, 1])

with r1:
    corr = filtered_df[["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]].corr()
    
    # 1. Matikan text_auto bawaan agar tidak bentrok atau error
    fig = px.imshow(
        corr,
        text_auto=False, # Dinonaktifkan karena kita akan buat teks manual yang aman
        title="Correlation Heatmap",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1
    )
    
    # 2. Jalankan fungsi pembersih Anda terlebih dahulu
    fig = clean_plotly(fig, 420)
    
    # 3. BUAT TEKS MANUAL DENGAN LOGIKAL WARNA KONTRAS
    annotations = []
    for i, row_name in enumerate(corr.index):
        for j, col_name in enumerate(corr.columns):
            val = corr.iloc[i, j]
            
            # Jika nilai kuat (kotak gelap), beri teks putih. Jika lemah, beri hitam.
            font_color = "white" if abs(val) > 0.4 else "black"
            
            annotations.append(
                dict(
                    x=col_name,
                    y=row_name,
                    text=f"{val:.2f}", # Membatasi 2 angka di belakang koma (termasuk 1.00)
                    showarrow=False,
                    font=dict(
                        color=font_color,
                        weight="bold",  # Membuat teks tebal agar terbaca jelas
                        size=13
                    )
                )
            )
            
    # 4. Terapkan teks manual tersebut ke dalam grafik
    fig.update_layout(annotations=annotations)
    
    st.plotly_chart(fig, use_container_width=True)

    chart_explanation(
        "This heatmap shows the correlation between numerical features. Values close to 1 mean a strong positive relationship, while values close to -1 mean a negative relationship.",
        "Tenure and TotalCharges usually have a strong relationship because customers who stay longer tend to have higher total charges."
    )

with r2:
    box_feature = st.selectbox(
        "Choose feature for boxplot",
        ["tenure", "MonthlyCharges", "TotalCharges"],
        index=1
    )
    fig = px.box(
        filtered_df,
        x="Churn",
        y=box_feature,
        color="Churn",
        title=f"{box_feature} vs Churn",
        color_discrete_map={"Yes": "#FF6B6B", "No": "#1971C2"}
    )
    st.plotly_chart(clean_plotly(fig, 420), use_container_width=True)

    chart_explanation(
        "This boxplot compares the selected numerical feature between churn and non-churn customers.",
        "If the box positions are different, the selected feature may have a relationship with churn behavior."
    )

# =========================================================
# CHURN ANALYSIS BY CATEGORICAL FEATURES
# =========================================================
st.markdown('<div class="section-title">③ Churn Analysis by Customer Profile</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-caption">Churn comparison by Dependents, Contract type, and Payment Method.</div>',
    unsafe_allow_html=True
)

cat1, cat2, cat3 = st.columns([1, 1, 2])

with cat1:
    dep = filtered_df.groupby(["Dependents", "Churn"], as_index=False).size()
    fig = px.bar(
        dep,
        x="Dependents",
        y="size",
        color="Churn",
        barmode="group",
        text="size",
        title="Churn Based on Dependents",
        color_discrete_map={"Yes": "#E8756E", "No": "#3B73B9"},
        labels={"size": "Customers", "Churn": ""}
    )
    fig.update_traces(textposition="outside", cliponaxis=False, textfont=dict(size=11))
    fig.update_layout(
        height=370,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=10, r=10, t=50, b=100),
        font=dict(family="Inter", color="#23324A", size=12),
        title_font=dict(size=15, color="#23324A"),
        bargap=0.25,
        legend=dict(
            title="",
            orientation="h",
            x=0.5, y=-0.18,
            xanchor="center",
            yanchor="top",
            font=dict(size=11, color="#23324A")
        ),
       xaxis=dict(showgrid=False, zeroline=False, title=""),
        yaxis=dict(showgrid=True, gridcolor="#E7DDCF", zeroline=False, title="Customers"),
    )
    st.plotly_chart(fig, use_container_width=True)

    chart_explanation(
        "This chart compares churn status between customers who have dependents and customers who do not have dependents.",
        "Customer profile features such as dependents can help explain differences in churn behavior."
    )

with cat2:
    contract = filtered_df.groupby(["Contract", "Churn"], as_index=False).size()
    fig = px.bar(
        contract,
        x="Contract",
        y="size",
        color="Churn",
        barmode="group",
        text="size",
        title="Churn Based on Contract",
        color_discrete_map={"Yes": "#E8756E", "No": "#3B73B9"},
        labels={"size": "Customers", "Churn": ""}
    )
    fig.update_traces(textposition="outside", cliponaxis=False, textfont=dict(size=11))
    fig.update_layout(
        height=370,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=10, r=10, t=50, b=100),
        font=dict(family="Inter", color="#23324A", size=12),
        title_font=dict(size=15, color="#23324A"),
        bargap=0.25,
        legend=dict(
            title="",
            orientation="h",
            x=0.5, y=-0.18,
            xanchor="center",
            yanchor="top",
            font=dict(size=11, color="#23324A")
        ),
        xaxis=dict(
    showgrid=False, zeroline=False, 
    title="",           # kosongkan, sudah jelas dari judul chart
    tickangle=-15, tickfont=dict(size=11)
),
        yaxis=dict(showgrid=True, gridcolor="#E7DDCF", zeroline=False, title="Customers"),
    )
    st.plotly_chart(fig, use_container_width=True)

    chart_explanation(
        "This chart shows customer churn based on contract type: month-to-month, one year, and two year.",
        "Contract type is an important churn indicator because customers with shorter contracts can leave more easily."
    )

with cat3:
    pay_rate = make_churn_rate_table(filtered_df, "PaymentMethod").sort_values("Churn Rate", ascending=True)
    fig = px.bar(
        pay_rate,
        y="PaymentMethod",
        x="Churn Rate",
        orientation="h",
        text=pay_rate["Churn Rate"].map(lambda x: f"{x:.1f}%"),
        title="Churn Rate by Payment Method",
        color="Churn Rate",
        color_continuous_scale=["#DCFCE7", "#FEF3C7", "#FEE2E2"],
        labels={"PaymentMethod": ""}
    )
    fig.update_traces(textposition="outside", cliponaxis=False, textfont=dict(size=11))
    fig.update_layout(
        height=370,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=10, r=10, t=50, b=10),
        font=dict(family="Inter", color="#23324A", size=12),
        title_font=dict(size=15, color="#23324A"),
        coloraxis=dict(
            colorbar=dict(
                title=dict(text="Churn Rate", font=dict(size=12, color="#23324A")),
                thickness=12,
                len=0.6,
                x=1.02,
                y=0.5,
                tickfont=dict(size=10, color="#23324A"),
                outlinewidth=0,
            )
        ),
        xaxis=dict(showgrid=True, gridcolor="#E7DDCF", zeroline=False, title="Churn Rate (%)"),
        yaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=11)),
    )
    st.plotly_chart(fig, use_container_width=True)

    chart_explanation(
        "This horizontal bar chart shows the churn rate for each payment method.",
        "Payment method can be related to churn risk, so companies can pay more attention to payment groups with higher churn rates."
    )

# =========================================================
# MODEL VISUALIZATION
# =========================================================
st.markdown('<div class="section-title">④ Model Visualization</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-caption">Logistic Regression sigmoid visualization and customer behavior scatter plot.</div>',
    unsafe_allow_html=True
)

mvis1, mvis2 = st.columns([1, 1])

with mvis1:
    sig_df = df[["TotalCharges", "Churn"]].copy()
    sig_df["ChurnNumeric"] = sig_df["Churn"].map({"Yes": 1, "No": 0})

    sigmoid_model = LogisticRegression(max_iter=1000)
    sigmoid_model.fit(sig_df[["TotalCharges"]], sig_df["ChurnNumeric"])

    x_range = np.linspace(sig_df["TotalCharges"].min(), sig_df["TotalCharges"].max(), 200)
    y_prob = sigmoid_model.predict_proba(pd.DataFrame({"TotalCharges": x_range}))[:, 1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_range,
        y=y_prob,
        mode="lines",
        name="Sigmoid Probability",
        line=dict(width=4, color="#FF6B6B")
    ))
    fig.add_trace(go.Scatter(
        x=sig_df["TotalCharges"],
        y=sig_df["ChurnNumeric"],
        mode="markers",
        name="Customers",
        marker=dict(size=5, opacity=0.25, color="#1971C2")
    ))
    fig.update_layout(
        title="Logistic Regression Sigmoid Line",
        xaxis_title="Total Charges",
        yaxis_title="Churn Probability"
    )
    st.plotly_chart(clean_plotly(fig, 420), use_container_width=True)

    chart_explanation(
        "This chart visualizes the Logistic Regression sigmoid line using TotalCharges as the input feature.",
        "The sigmoid curve helps explain probability-based prediction, but one feature alone is not enough to fully explain churn."
    )

with mvis2:
    fig = px.scatter(
        filtered_df,
        x="MonthlyCharges",
        y="TotalCharges",
        size="tenure",
        color="Churn",
        hover_data=["customerID", "Contract", "InternetService", "PaymentMethod"],
        title="Monthly Charges vs Total Charges",
        color_discrete_map={"Yes": "#FF6B6B", "No": "#1971C2"}
    )
    st.plotly_chart(clean_plotly(fig, 420), use_container_width=True)

    chart_explanation(
        "This scatter plot compares MonthlyCharges and TotalCharges. Bubble size represents tenure, and color represents churn status.",
        "This visualization helps show customer behavior patterns based on payment amount, total spending, and subscription duration."
    )

# =========================================================
# MACHINE LEARNING MODEL PERFORMANCE
# =========================================================
st.markdown('<div class="section-title">⑤ Machine Learning Model Performance</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-caption">Latest notebook result: comparison of Logistic Regression, K-NN, Naive Bayes, and Random Forest using Non-SMOTE and SMOTE training.</div>',
    unsafe_allow_html=True
)

tabs = st.tabs(["Model Metrics", "Confusion Matrix", "SMOTE & Columns", "Best Params", "Raw Data Preview"])

metrics_df = metrics_dataframe(models)
metrics_sorted = metrics_df.sort_values(["Type", "Accuracy"], ascending=[True, False])

with tabs[0]:
    summary1, summary2 = st.columns([1.15, 1])

    with summary1:
        st.dataframe(
            metrics_sorted.style.format({
                "Accuracy": "{:.2%}",
                "Precision": "{:.2%}",
                "Recall": "{:.2%}",
                "F1-Score": "{:.2%}",
            }),
            use_container_width=True,
            height=330
        )

    with summary2:
        metric_choice = st.selectbox(
            "Choose metric to compare",
            ["Accuracy", "Precision", "Recall", "F1-Score"],
            index=0
        )

        chart_df = metrics_df.sort_values(metric_choice, ascending=False).copy()

        chart_df["Model Display"] = chart_df["Model"].replace({
            "Logistic Regression": "Logistic Regression",
            "Logistic Regression (SMOTE)": "Logistic Regression (SMOTE)",
            "K-Nearest Neighbors": "K-Nearest Neighbors",
            "K-Nearest Neighbors (SMOTE)": "K-Nearest Neighbors (SMOTE)",
            "Naive Bayes": "Naive Bayes",
            "Naive Bayes (SMOTE)": "Naive Bayes (SMOTE)",
            "Random Forest": "Random Forest",
            "Random Forest (SMOTE)": "Random Forest (SMOTE)",
        })

        fig = px.bar(
            chart_df,
            x="Model Display",
            y=metric_choice,
            color="Type",
            text=chart_df[metric_choice].map(lambda x: f"{x:.1%}"),
            title=f"{metric_choice} Comparison",
            color_discrete_map={
                "Non-SMOTE": "#3B73B9",
                "SMOTE": "#E8756E"
            },
            hover_data={
                "Model": True,
                "Model Display": False,
                metric_choice: ":.2%",
                "Type": True
            }
        )

        fig.update_yaxes(tickformat=".0%", title_text="")
        fig.update_xaxes(title_text="", tickangle=-30, tickfont=dict(size=11))
        fig.update_traces(
            textposition="outside",
            textfont=dict(size=10, color="#23324A"),
            cliponaxis=False
        )

        fig.update_layout(
            height=330,
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin=dict(l=10, r=10, t=50, b=60),
            font=dict(family="Inter", color="#23324A", size=12),
            title_font=dict(size=16, color="#23324A"),
            bargap=0.2,
            uniformtext_minsize=8,
            uniformtext_mode="hide",
            legend=dict(
                title="",
                orientation="h",
                x=0.5,
                y=1.12,
                xanchor="center",
                yanchor="top",
                bgcolor="rgba(0,0,0,0)",
                font=dict(size=11, color="#23324A"),
            )
        )

        fig.update_xaxes(showgrid=True, gridcolor="#E7DDCF", zeroline=False)
        fig.update_yaxes(showgrid=True, gridcolor="#E7DDCF", zeroline=False)

        st.plotly_chart(fig, use_container_width=True)

        chart_explanation(
            "This bar chart compares machine learning models using the selected metric, such as Accuracy, Precision, Recall, or F1-Score.",
            "The best model depends on the chosen metric. For churn prediction, recall is important because the company wants to detect as many churn customers as possible."
        )

    best_acc = metrics_df.sort_values("Accuracy", ascending=False).iloc[0]
    best_recall = metrics_df.sort_values("Recall", ascending=False).iloc[0]

    b1, b2 = st.columns(2)

    with b1:
        st.success(f"Highest Accuracy: {best_acc['Model']} — {best_acc['Accuracy']:.2%}")

    with b2:
        st.info(f"Highest Churn Recall: {best_recall['Model']} — {best_recall['Recall']:.2%}")

with tabs[1]:
    selected_model = st.selectbox("Choose model", list(models.keys()))
    cm = models[selected_model]["cm"]

    fig = px.imshow(
        cm,
        text_auto=True,
        labels=dict(x="Predicted", y="Actual"),
        x=["No Churn", "Churn"],
        y=["No Churn", "Churn"],
        title=f"Confusion Matrix — {selected_model}",
        color_continuous_scale="Blues"
    )
    st.plotly_chart(clean_plotly(fig, 430), use_container_width=True)

    chart_explanation(
        "The confusion matrix compares actual churn status and predicted churn status. It shows correct and incorrect predictions.",
        "This chart helps evaluate model mistakes, especially false negatives where churn customers are predicted as non-churn."
    )

    tn, fp, fn, tp = cm.ravel()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("True Negative", f"{tn:,}")
    c2.metric("False Positive", f"{fp:,}")
    c3.metric("False Negative", f"{fn:,}")
    c4.metric("True Positive", f"{tp:,}")

with tabs[2]:
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown("#### Before SMOTE")
        st.json(model_info["before_smote"])
    with s2:
        st.markdown("#### After SMOTE")
        st.json(model_info["after_smote"])
    with s3:
        st.markdown("#### Test Distribution")
        st.json(model_info["test_distribution"])

    st.markdown("#### Remaining Feature Columns After VIF Removal")
    st.write(model_info["prediction_columns"])

    st.markdown("#### Dropped Multicollinearity Columns")
    st.write(model_info["dropped_columns"])

with tabs[3]:
    param_table = pd.DataFrame([
        {
            "Model": name,
            "Type": info["mode"],
            "Best Params": str(info["best_params"]),
        }
        for name, info in models.items()
    ])
    st.dataframe(param_table, use_container_width=True, height=380)

with tabs[4]:
    st.dataframe(filtered_df, use_container_width=True, height=420)
    st.download_button(
        "Download filtered CSV",
        filtered_df.to_csv(index=False).encode("utf-8"),
        "filtered_telco_churn.csv",
        "text/csv"
    )

# =========================================================
# SINGLE CUSTOMER PREDICTION
# =========================================================
st.markdown("---")
st.markdown('<div class="section-title">🔮 Single Customer Churn Prediction</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-caption">Input follows the 13 remaining feature columns after multicollinearity removal.</div>',
    unsafe_allow_html=True
)

with st.form("prediction_form"):
    f1, f2, f3 = st.columns(3)

    with f1:
        selected_prediction_model = st.selectbox(
            "Prediction Model",
            list(models.keys()),
            index=list(models.keys()).index("K-Nearest Neighbors") if "K-Nearest Neighbors" in models else 0
        )
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])

    with f2:
        tenure = st.slider("Tenure", 0, 72, 12)
        phone = st.selectbox("Phone Service", ["No", "Yes"])
        paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
        monthly = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)

    with f3:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        payment = st.selectbox(
            "Payment Method",
            [
                "Bank transfer (automatic)",
                "Credit card (automatic)",
                "Electronic check",
                "Mailed check",
            ]
        )

    submitted = st.form_submit_button("Predict Churn Risk")

if submitted:
    total = monthly * max(tenure, 1)
    scaled_values = model_info["scaler"].transform(
        pd.DataFrame(
            [[tenure, monthly, total]],
            columns=["tenure", "MonthlyCharges", "TotalCharges"]
        )
    )
    scaled_tenure = scaled_values[0][0]
    scaled_monthly = scaled_values[0][1]

    input_data = pd.DataFrame([{
        "gender": 1 if gender == "Male" else 0,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": 1 if partner == "Yes" else 0,
        "Dependents": 1 if dependents == "Yes" else 0,
        "tenure": scaled_tenure,
        "PhoneService": 1 if phone == "Yes" else 0,
        "PaperlessBilling": 1 if paperless == "Yes" else 0,
        "MonthlyCharges": scaled_monthly,
        "Contract_One year": 1 if contract == "One year" else 0,
        "Contract_Two year": 1 if contract == "Two year" else 0,
        "PaymentMethod_Credit card (automatic)": 1 if payment == "Credit card (automatic)" else 0,
        "PaymentMethod_Electronic check": 1 if payment == "Electronic check" else 0,
        "PaymentMethod_Mailed check": 1 if payment == "Mailed check" else 0,
    }])

    input_data = input_data[model_info["prediction_columns"]]

    selected_model_object = models[selected_prediction_model]["model"]
    prediction = int(selected_model_object.predict(input_data)[0])

    if hasattr(selected_model_object, "predict_proba"):
        probability = selected_model_object.predict_proba(input_data)[0][1]
    else:
        probability = np.nan

    if prediction == 1:
        st.error(f"⚠️ Customer is likely to churn. Churn probability: {probability:.2%}")
    else:
        st.success(f"✅ Customer is not likely to churn. Churn probability: {probability:.2%}")

    st.markdown("#### Encoded Input Preview")
    st.dataframe(input_data, use_container_width=True)

    st.caption("Note: tenure and MonthlyCharges are standardized using the same scaler as the training data.")

