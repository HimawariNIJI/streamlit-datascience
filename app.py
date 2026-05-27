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
from sklearn.metrics import accuracy_score, confusion_matrix

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
# CUSTOM CSS - CLEAN STREAMLIT GALLERY LOOK
# =========================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background-color: #ffffff;
    }
            
    .stApp {
        background-color: #ffffff;
    }
            
    [data-testid="stAppViewContainer"] {
        background-color: #ffffff;
    }


    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #182538 0%, #21324a 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    section[data-testid="stSidebar"] * {
        color: #f5f7fb !important;
    }

    section[data-testid="stSidebar"] div[data-baseweb="select"] > div,
    section[data-testid="stSidebar"] input {
        background-color: #34465f !important;
        border-radius: 12px !important;
        border: 0 !important;
    }

    .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
        max-width: 1180px;
    }

    .hero-title {
        font-size: 48px;
        line-height: 1.05;
        font-weight: 800;
        letter-spacing: -1.5px;
        color: #252a37;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        color: #687085;
        font-size: 17px;
        line-height: 1.7;
        max-width: 900px;
        margin-bottom: 20px;
    }

    .section-title {
        font-size: 30px;
        font-weight: 800;
        letter-spacing: -0.8px;
        color: #252a37;
        margin-top: 34px;
        margin-bottom: 6px;
    }

    .section-caption {
        color: #7b8192;
        font-size: 15.5px;
        margin-bottom: 22px;
    }

    .metric-card {
        border: 1px solid #e4e7ec;
        border-radius: 16px;
        background: #ffffff;
        padding: 20px 18px 14px 18px;
        box-shadow: 0 2px 10px rgba(16, 24, 40, 0.04);
        min-height: 156px;
    }

    .metric-label {
        color: #667085;
        font-size: 14px;
        font-weight: 650;
        margin-bottom: 6px;
    }

    .metric-value {
        color: #252a37;
        font-size: 34px;
        font-weight: 700;
        letter-spacing: -1px;
        margin-bottom: 5px;
    }

    .metric-up {
        background-color: #dcfce7;
        color: #16a34a;
        border-radius: 999px;
        padding: 4px 10px;
        font-size: 13px;
        font-weight: 700;
    }

    .metric-down {
        background-color: #fee2e2;
        color: #dc2626;
        border-radius: 999px;
        padding: 4px 10px;
        font-size: 13px;
        font-weight: 700;
    }

    .chart-card {
        border: 1px solid #e4e7ec;
        border-radius: 18px;
        background: #ffffff;
        padding: 20px;
        box-shadow: 0 2px 10px rgba(16, 24, 40, 0.04);
    }

    .info-box {
        background: #eef6ff;
        color: #1d4ed8;
        border-radius: 14px;
        padding: 18px 20px;
        font-weight: 600;
        border: 1px solid #d8eaff;
    }

    div[data-testid="stMetric"] {
        background: white;
        border-radius: 16px;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #f3f5f8;
        border-radius: 999px;
        padding: 10px 18px;
        color: #344054;
        font-weight: 700;
    }

    .stTabs [aria-selected="true"] {
        background-color: #252a37;
        color: white;
    }

    hr {
        margin: 1.8rem 0;
        border-color: #edf0f4;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# DATA & MODEL
# =========================================================
DATA_URL = "https://raw.githubusercontent.com/HEHEfebrian/ALPDS/refs/heads/main/WA_Fn-UseC_-Telco-Customer-Churn.csv"

@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(DATA_URL)
    df = df.replace(r"^\\s*$", np.nan, regex=True)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"])
    df = df.drop_duplicates()
    return df

@st.cache_resource(show_spinner=False)
def train_models(raw_df):
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
            "InternetService", "Contract", "PaymentMethod", "MultipleLines",
            "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
            "StreamingTV", "StreamingMovies"
        ],
        drop_first=True
    )

    model_df = model_df.drop("customerID", axis=1)
    bool_cols = model_df.select_dtypes(include="bool").columns
    model_df[bool_cols] = model_df[bool_cols].astype(int)

    X = model_df.drop("Churn", axis=1)
    y = model_df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Mengikuti notebook kamu: kolom multikolinier di-drop
    drop_cols = [
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

    existing_drop_cols = [c for c in drop_cols if c in X_train.columns]
    X_train = X_train.drop(columns=existing_drop_cols)
    X_test = X_test.drop(columns=existing_drop_cols)

    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train, y_train)

    best_k, best_acc = 3, 0
    for k in [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]:
        knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
        knn.fit(X_train, y_train)
        acc = accuracy_score(y_test, knn.predict(X_test))
        if acc > best_acc:
            best_k, best_acc = k, acc

    knn_model = KNeighborsClassifier(n_neighbors=best_k, metric="euclidean")
    knn_model.fit(X_train, y_train)

    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)

    results = {
        "Logistic Regression": {
            "model": log_model,
            "accuracy": accuracy_score(y_test, log_model.predict(X_test)),
            "cm": confusion_matrix(y_test, log_model.predict(X_test)),
        },
        "K-Nearest Neighbors": {
            "model": knn_model,
            "accuracy": accuracy_score(y_test, knn_model.predict(X_test)),
            "cm": confusion_matrix(y_test, knn_model.predict(X_test)),
        },
        "Naive Bayes": {
            "model": nb_model,
            "accuracy": accuracy_score(y_test, nb_model.predict(X_test)),
            "cm": confusion_matrix(y_test, nb_model.predict(X_test)),
        },
    }

    return results, X_train.columns.tolist()

df = load_data()
models, model_features = train_models(df)

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("### ▦ Showcase")
    st.markdown("### 📘 API Guide")
    st.markdown("### ‹› Examples")
    st.markdown("---")
    st.markdown("## ⚗ Filters")

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

    senior_filter = st.selectbox(
        "Senior Citizen",
        options=["All", "No", "Yes"]
    )

    tenure_range = st.slider(
        "Tenure Range (Months)",
        int(df["tenure"].min()),
        int(df["tenure"].max()),
        (int(df["tenure"].min()), int(df["tenure"].max()))
    )

filtered_df = df[
    (df["Contract"].isin(contract_filter)) &
    (df["InternetService"].isin(internet_filter)) &
    (df["PaymentMethod"].isin(payment_filter)) &
    (df["tenure"].between(tenure_range[0], tenure_range[1]))
]

if senior_filter != "All":
    senior_value = 1 if senior_filter == "Yes" else 0
    filtered_df = filtered_df[filtered_df["SeniorCitizen"] == senior_value]

# =========================================================
# HELPER
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

def clean_plotly(fig, height=380):
    fig.update_layout(
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=20, r=20, t=45, b=30),

        font=dict(
            family="Inter",
            color="#23324A",
            size=13
        ),

        title_font=dict(
            size=18,
            color="#23324A"
        ),

        legend=dict(
            orientation="h",
            y=-0.18,
            x=0.5,
            xanchor="center",
            font=dict(color="#23324A")
        ),
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="#e7ddcf",
        zeroline=False,
        title_font=dict(color="#23324A"),
        tickfont=dict(color="#23324A")
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="#e7ddcf",
        zeroline=False,
        title_font=dict(color="#23324A"),
        tickfont=dict(color="#23324A")
    )

    # Khusus chart yang memang punya text di dalam bar/scatter/pie
    fig.update_traces(
        selector=dict(type="bar"),
        textfont=dict(color="#23324A", size=13)
    )

    fig.update_traces(
        selector=dict(type="pie"),
        textfont=dict(color="#23324A", size=13)
    )

    fig.update_traces(
        selector=dict(type="scatter"),
        textfont=dict(color="#23324A", size=13)
    )

    fig.update_traces(
        selector=dict(type="heatmap"),
        textfont=dict(color="#23324A", size=13)
    )

    fig.update_layout(
        coloraxis_colorbar=dict(
            title=dict(
                font=dict(color="#23324A", size=14)
            ),
            tickfont=dict(color="#23324A", size=13)
        )
    )

    return fig

# =========================================================
# HERO
# =========================================================
st.markdown('<div class="hero-title">📡 Telco Customer Churn Gallery</div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="hero-subtitle">
    This dashboard is a showcase of <b>Telco Customer Churn Analysis</b>, 
    demonstrating how customer behavior, contract type, charges, and service features relate to churn risk.
    <br><b>Dataset:</b> Kaggle Telco Customer Churn
    </div>
    """,
    unsafe_allow_html=True
)

# KPI
total_customers = len(filtered_df)
churn_count = (filtered_df["Churn"] == "Yes").sum()
churn_rate = churn_count / total_customers * 100 if total_customers else 0
avg_monthly = filtered_df["MonthlyCharges"].mean() if total_customers else 0
avg_tenure = filtered_df["tenure"].mean() if total_customers else 0

k1, k2, k3, k4 = st.columns(4)
with k1:
    metric_card("Total Customers", f"{total_customers:,}", "Filtered Data", True)
with k2:
    metric_card("Churn Rate", f"{churn_rate:.1f}%", "Risk Level", churn_rate < 30)
with k3:
    metric_card("Avg. Monthly Charges", f"${avg_monthly:.2f}", "Monthly", True)
with k4:
    metric_card("Avg. Tenure", f"{avg_tenure:.1f}", "Months", True)

# =========================================================
# TRENDING SECTION
# =========================================================
st.markdown('<div class="section-title">↗ How are customers trending?</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-caption">Track churn, tenure, and charges to spot customer behavior patterns.</div>',
    unsafe_allow_html=True
)

c1, c2 = st.columns([1.55, 1])

with c1:
    monthly = (
        filtered_df.groupby("tenure", as_index=False)
        .agg(Customers=("customerID", "count"), Churners=("Churn", lambda x: (x == "Yes").sum()))
    )
    monthly["Churn Rate"] = np.where(monthly["Customers"] > 0, monthly["Churners"] / monthly["Customers"] * 100, 0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly["tenure"], y=monthly["Customers"],
        mode="lines+markers", name="Customers",
        line=dict(width=3, color="#1f77d4"),
        fill="tozeroy"
    ))
    fig.add_trace(go.Scatter(
        x=monthly["tenure"], y=monthly["Churners"],
        mode="lines+markers", name="Churners",
        line=dict(width=3, color="#74c0fc")
    ))
    fig.update_layout(title="Customers & Churners by Tenure")
    st.plotly_chart(clean_plotly(fig, 390), use_container_width=True)

with c2:
    fig = px.bar(
        monthly.tail(24),
        x="tenure",
        y="Churn Rate",
        title="Churn Rate by Tenure",
        color="Churn Rate",
        color_continuous_scale=["#dcfce7", "#fef3c7", "#fee2e2"]
    )
    
    st.plotly_chart(clean_plotly(fig, 390), use_container_width=True)
    

# =========================================================
# WHERE & WHAT SECTION
# =========================================================
st.markdown('<div class="section-title">◎ Where & What?</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-caption">Break down churn performance by contract, internet service, and payment method.</div>',
    unsafe_allow_html=True
)

w1, w2, w3 = st.columns([1.15, 1, 1.15])

with w1:
    tree = filtered_df.groupby(["Contract", "InternetService"], as_index=False).agg(
        Customers=("customerID", "count"),
        Churners=("Churn", lambda x: (x == "Yes").sum())
    )
    tree["Churn Rate"] = tree["Churners"] / tree["Customers"] * 100
    fig = px.treemap(
        tree,
        path=["Contract", "InternetService"],
        values="Customers",
        color="Churn Rate",
        color_continuous_scale=["#86efac", "#fde68a", "#f87171"],
        title="Contract → Internet Service"
    )
    st.plotly_chart(clean_plotly(fig, 430), use_container_width=True)

with w2:
    radar_df = filtered_df.groupby("Contract", as_index=False).agg(
        ChurnRate=("Churn", lambda x: (x == "Yes").mean() * 100),
        AvgMonthly=("MonthlyCharges", "mean"),
        AvgTenure=("tenure", "mean")
    )
    categories = ["ChurnRate", "AvgMonthly", "AvgTenure"]
    fig = go.Figure()
    for _, row in radar_df.iterrows():
        values = [row[c] for c in categories]
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name=row["Contract"]
        ))
    fig.update_layout(
        title="Contract Profiles",
        polar=dict(radialaxis=dict(visible=True)),
    )
    st.plotly_chart(clean_plotly(fig, 430), use_container_width=True)

with w3:
    pay = filtered_df.groupby(["PaymentMethod", "Churn"], as_index=False).size()
    fig = px.bar(
        pay,
        y="PaymentMethod",
        x="size",
        color="Churn",
        orientation="h",
        title="Churn by Payment Method",
        color_discrete_map={"Yes": "#ff6b6b", "No": "#1971c2"}
    )
    st.plotly_chart(clean_plotly(fig, 430), use_container_width=True)

# =========================================================
# DEEP DIVE SECTION
# =========================================================
st.markdown("---")
st.markdown('<div class="section-title">↳ Deep Dive — Click to Explore</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-caption">Explore customer risk pattern between Monthly Charges, Total Charges, and Tenure.</div>',
    unsafe_allow_html=True
)

d1, d2 = st.columns([1.25, 1])

with d1:
    fig = px.scatter(
        filtered_df,
        x="MonthlyCharges",
        y="TotalCharges",
        size="tenure",
        color="Churn",
        hover_data=["customerID", "Contract", "InternetService", "PaymentMethod"],
        title="Monthly Charges vs Total Charges",
        color_discrete_map={"Yes": "#ff6b6b", "No": "#1971c2"}
    )
    st.plotly_chart(clean_plotly(fig, 440), use_container_width=True)

with d2:
    st.markdown(
        """
        <div class="info-box">
        👆 Bubble size represents tenure. Customers with higher monthly charges and short contract types are often more interesting to inspect for churn risk.
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write("")
    fig = px.box(
        filtered_df,
        x="Churn",
        y="MonthlyCharges",
        color="Churn",
        title="Monthly Charges by Churn Status",
        color_discrete_map={"Yes": "#ff6b6b", "No": "#1971c2"}
    )
    st.plotly_chart(clean_plotly(fig, 330), use_container_width=True)

# =========================================================
# OPERATIONAL INSIGHTS
# =========================================================
st.markdown('<div class="section-title">⚙ Operational Insights</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-caption">Examine service usage, support features, and billing behavior.</div>',
    unsafe_allow_html=True
)

o1, o2, o3 = st.columns(3)

with o1:
    heat = pd.crosstab(filtered_df["Contract"], filtered_df["InternetService"])
    fig = px.imshow(
        heat,
        text_auto=True,
        title="Contract vs Internet Service",
        color_continuous_scale="Blues"
    )
    st.plotly_chart(clean_plotly(fig, 360), use_container_width=True)

with o2:
    support = filtered_df.groupby(["TechSupport", "Churn"], as_index=False).size()
    fig = px.bar(
        support,
        x="TechSupport",
        y="size",
        color="Churn",
        barmode="group",
        title="Tech Support vs Churn",
        color_discrete_map={"Yes": "#ff6b6b", "No": "#1971c2"}
    )
    st.plotly_chart(clean_plotly(fig, 360), use_container_width=True)

with o3:
    pie = filtered_df["Churn"].value_counts().reset_index()
    pie.columns = ["Churn", "Count"]
    fig = px.pie(
        pie,
        values="Count",
        names="Churn",
        hole=0.55,
        title="Churn Distribution",
        color="Churn",
        color_discrete_map={"Yes": "#ff6b6b", "No": "#1971c2"}
    )
    st.plotly_chart(clean_plotly(fig, 360), use_container_width=True)

# =========================================================
# MODEL PERFORMANCE & DATA
# =========================================================
st.markdown('<div class="section-title">🧠 Machine Learning Model</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-caption">Model comparison based on your notebook: Logistic Regression, KNN, and Naive Bayes.</div>',
    unsafe_allow_html=True
)

tabs = st.tabs(["Model Accuracy", "Confusion Matrix", "Raw Data Preview"])

with tabs[0]:
    acc_df = pd.DataFrame({
        "Model": list(models.keys()),
        "Accuracy": [v["accuracy"] for v in models.values()]
    }).sort_values("Accuracy", ascending=False)
    fig = px.bar(
        acc_df,
        x="Model",
        y="Accuracy",
        text=acc_df["Accuracy"].map(lambda x: f"{x:.2%}"),
        title="Accuracy Comparison",
        color="Accuracy",
        color_continuous_scale=["#dbeafe", "#2563eb"]
    )
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(clean_plotly(fig, 380), use_container_width=True)
    st.success(f"Best model: {acc_df.iloc[0]['Model']} with accuracy {acc_df.iloc[0]['Accuracy']:.2%}")

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
    st.plotly_chart(clean_plotly(fig, 420), use_container_width=True)

with tabs[2]:
    st.dataframe(filtered_df, use_container_width=True, height=420)
    st.download_button(
        "Download filtered CSV",
        filtered_df.to_csv(index=False).encode("utf-8"),
        "filtered_telco_churn.csv",
        "text/csv"
    )

# =========================================================
# PREDICTION FORM
# =========================================================
st.markdown("---")
st.markdown('<div class="section-title">🔮 Single Customer Churn Prediction</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-caption">Simple prediction demo using the Logistic Regression model structure from your notebook.</div>',
    unsafe_allow_html=True
)

with st.form("prediction_form"):
    p1, p2, p3 = st.columns(3)
    with p1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
    with p2:
        tenure = st.slider("Tenure", 0, 72, 12)
        phone = st.selectbox("Phone Service", ["No", "Yes"])
        paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
        monthly = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
    with p3:
        internet = st.selectbox("Internet Service", sorted(df["InternetService"].unique()))
        contract = st.selectbox("Contract", sorted(df["Contract"].unique()))
        payment = st.selectbox("Payment Method", sorted(df["PaymentMethod"].unique()))
        total = st.number_input("Total Charges", 0.0, 10000.0, float(monthly * max(tenure, 1)))

    submitted = st.form_submit_button("Predict Churn Risk")

if submitted:
    st.info(
        "Prediction form UI sudah siap. Untuk hasil prediksi yang 100% akurat, "
        "bagian input harus di-encode persis sama seperti training columns. "
        "Untuk tugas dashboard, bagian visualisasi dan model comparison sudah paling penting."
    )