import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.base import clone, BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

st.set_page_config(
    page_title="Laptop Price Prediction",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded",
)

pio.templates.default = "plotly_dark"


def style_plotly(fig):
    fig.update_layout(
        paper_bgcolor="#16161D",
        plot_bgcolor="#16161D",
        font=dict(color="#FFFFFF"),
        title_font=dict(color="#FFFFFF"),
        legend=dict(font=dict(color="#FFFFFF")),
        margin=dict(l=20, r=20, t=70, b=40),
        xaxis=dict(
            gridcolor="#2A2A35",
            zerolinecolor="#2A2A35",
            tickfont=dict(color="#D1D5DB"),
            title_font=dict(color="#FFFFFF"),
        ),
        yaxis=dict(
            gridcolor="#2A2A35",
            zerolinecolor="#2A2A35",
            tickfont=dict(color="#D1D5DB"),
            title_font=dict(color="#FFFFFF"),
        ),
    )

    if hasattr(fig.layout, "coloraxis") and fig.layout.coloraxis is not None:
        fig.update_layout(
            coloraxis_colorbar=dict(
                tickfont=dict(color="#D1D5DB"),
                title_font=dict(color="#FFFFFF"),
            )
        )

    return fig


CUSTOM_CSS = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {background: transparent !important;}

    html, body, [class*="css"] {
        font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
    }

    [data-testid="stAppViewContainer"] {
        background-color: #0B0B0F !important;
        color: #FFFFFF !important;
    }

    .stApp {
        background-color: #0B0B0F !important;
        color: #FFFFFF !important;
    }

    .main {
        background-color: #0B0B0F !important;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        background-color: #0B0B0F !important;
    }

    [data-testid="stHeader"] {
        background: #0B0B0F !important;
    }

    [data-testid="stToolbar"] {
        right: 1rem;
    }

    section[data-testid="stSidebar"] {
        background-color: #0B0B0F !important;
        border-right: 1px solid #2A2A35;
    }

    section[data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
        font-weight: 800 !important;
    }

    p, span, label {
        color: #FFFFFF !important;
    }

    div, li {
        color: #D1D5DB;
    }

    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stCaptionContainer"],
    .stCaption {
        color: #D1D5DB !important;
    }

    [data-testid="stTabs"] {
        background-color: #0B0B0F !important;
    }

    [data-testid="stTabs"] button {
        color: #D1D5DB !important;
        background: transparent !important;
    }

    [data-testid="stTabs"] button[aria-selected="true"] {
        color: #E50914 !important;
        border-bottom: 2px solid #E50914 !important;
    }

    .stSelectbox label, .stSlider label, .stNumberInput label, .stTextInput label,
    .stFileUploader label, .stRadio label, .stMultiSelect label {
        color: #FFFFFF !important;
        font-weight: 700;
    }

    div[data-baseweb="select"] > div,
    div[data-testid="stNumberInput"] input,
    div[data-testid="stTextInput"] input,
    textarea {
        background-color: #16161D !important;
        color: #FFFFFF !important;
        border: 1px solid #2A2A35 !important;
    }

    .stButton > button, .stFormSubmitButton > button {
        background-color: #E50914 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px;
        font-weight: 800;
        padding: 0.65rem 1rem;
    }

    .stButton > button:hover, .stFormSubmitButton > button:hover {
        background-color: #B20710 !important;
        color: #FFFFFF !important;
    }

    div[data-testid="stMetric"] {
        background: #16161D;
        border: 1px solid #2A2A35;
        padding: 1rem;
        border-radius: 12px;
    }

    div[data-testid="stMetricLabel"] {
        color: #D1D5DB !important;
    }

    div[data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-weight: 800;
    }

    div[data-testid="stMetricDelta"] {
        color: #E50914 !important;
    }

    .hero {
        background: linear-gradient(135deg, #16161D 0%, #1F1F2A 60%, #2A0A0E 100%);
        padding: 2.8rem 2.6rem;
        border-radius: 16px;
        margin-bottom: 1.6rem;
        border: 1px solid #2A2A35;
        box-shadow: 0 10px 30px rgba(0,0,0,0.40);
    }

    .hero h1 {
        color: #FFFFFF !important;
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }

    .hero p {
        color: #D1D5DB !important;
        font-size: 1.05rem;
        max-width: 850px;
        line-height: 1.6;
    }

    .hero .tagline {
        display: inline-block;
        color: #E50914 !important;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        font-size: 0.82rem;
        margin-bottom: 0.8rem;
    }

    .kpi-card {
        background: #16161D;
        border: 1px solid #2A2A35;
        border-radius: 12px;
        padding: 1.25rem 1.35rem;
        text-align: left;
        box-shadow: 0 4px 14px rgba(0,0,0,0.35);
        min-height: 125px;
    }

    .kpi-label {
        color: #D1D5DB !important;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.35rem;
        font-weight: 700;
    }

    .kpi-value {
        color: #FFFFFF !important;
        font-size: 1.65rem;
        font-weight: 900;
        line-height: 1.2;
    }

    .kpi-sub {
        color: #E50914 !important;
        font-size: 0.8rem;
        margin-top: 0.35rem;
        font-weight: 700;
    }

    .section-card {
        background: #16161D;
        border: 1px solid #2A2A35;
        border-radius: 12px;
        padding: 1.5rem 1.6rem;
        margin-bottom: 1rem;
        color: #D1D5DB !important;
        line-height: 1.8;
        box-shadow: 0 4px 14px rgba(0,0,0,0.25);
    }

    .section-card * {
        color: #D1D5DB !important;
    }

    .section-card b {
        color: #FFFFFF !important;
    }

    .chart-note {
        color: #D1D5DB !important;
        font-size: 0.92rem;
        margin-top: 0.4rem;
        margin-bottom: 1rem;
        padding-left: 0.1rem;
    }

    .pred-card {
        background: linear-gradient(135deg, #1F1F2A 0%, #2A0A0E 100%);
        border-radius: 16px;
        padding: 2.2rem;
        text-align: center;
        border: 1px solid #E50914;
        box-shadow: 0 8px 25px rgba(229,9,20,0.20);
    }

    .pred-card .price {
        font-size: 2.8rem;
        font-weight: 900;
        color: #FFFFFF !important;
    }

    .pred-card .label {
        color: #D1D5DB !important;
        font-size: 0.9rem;
        margin-bottom: 0.6rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    .pred-card .modelname {
        color: #E50914 !important;
        font-weight: 800;
        font-size: 0.95rem;
        margin-top: 0.6rem;
    }

    .badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 800;
        margin-right: 0.4rem;
    }

    .badge-best {
        background: rgba(229, 9, 20, 0.15);
        color: #E50914 !important;
        border: 1px solid #E50914;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #2A2A35;
        border-radius: 10px;
        background-color: #16161D !important;
    }

    table {
        color: #FFFFFF !important;
    }

    hr {
        border-color: #2A2A35;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "laptopData.csv")


@st.cache_data(show_spinner=False)
def load_raw_data(source):
    return pd.read_csv(source)


def convert_memory_to_gb(value):
    value = str(value)
    parts = value.split("+")
    total = 0

    for part in parts:
        part = part.strip()

        if not part:
            continue

        size = part.split()[0]

        try:
            if "TB" in size:
                number = float(size.replace("TB", ""))
                total += number * 1024
            elif "GB" in size:
                number = float(size.replace("GB", ""))
                total += number
        except ValueError:
            continue

    return total


@st.cache_data(show_spinner=False)
def clean_data(df_raw):
    stats = {}
    df = df_raw.copy()

    required_columns = {"Ram", "Weight", "Inches", "Price"}
    missing_required = sorted(required_columns - set(df.columns))

    if missing_required:
        raise ValueError(f"Missing required columns in dataset: {missing_required}")

    stats["shape_before"] = df.shape
    stats["missing_before"] = df.isna().sum()

    if "Unnamed: 0" in df.columns:
        df = df.drop("Unnamed: 0", axis=1)

    stats["fully_empty_rows"] = int(df.isna().all(axis=1).sum())
    df = df.dropna(how="all")

    stats["duplicate_rows"] = int(df.duplicated().sum())
    df = df.drop_duplicates()

    df["Ram"] = df["Ram"].astype(str).str.replace("GB", "", regex=False)
    df["Ram"] = pd.to_numeric(df["Ram"], errors="coerce")

    df["Weight"] = df["Weight"].astype(str).str.replace("kg", "", regex=False)
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")
    df["Weight"] = df["Weight"].fillna(df["Weight"].median())

    df["Inches"] = pd.to_numeric(df["Inches"], errors="coerce")
    df["Inches"] = df["Inches"].fillna(df["Inches"].median())

    if "Memory" in df.columns:
        df["Memory_GB"] = df["Memory"].apply(convert_memory_to_gb)
        df = df.drop("Memory", axis=1)

    if "Cpu" in df.columns:
        df["Cpu_Brand"] = df["Cpu"].astype(str).str.split().str[0]
        df = df.drop("Cpu", axis=1)

    if "Gpu" in df.columns:
        df["Gpu_Brand"] = df["Gpu"].astype(str).str.split().str[0]
        df = df.drop("Gpu", axis=1)

    df = df.dropna(subset=["Price"])
    df["Ram"] = df["Ram"].fillna(df["Ram"].median())

    Q1 = df["Price"].quantile(0.25)
    Q3 = df["Price"].quantile(0.75)
    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    stats["outliers_removed"] = int(((df["Price"] < lower_limit) | (df["Price"] > upper_limit)).sum())
    df = df[(df["Price"] >= lower_limit) & (df["Price"] <= upper_limit)]

    stats["shape_after"] = df.shape
    stats["missing_after"] = df.isna().sum()

    return df.reset_index(drop=True), stats


def build_model_registry():
    return {
        "Linear Regression": {
            "model": LinearRegression(),
            "scaled": True,
            "family": "Linear",
        },
        "Ridge Regression": {
            "model": Ridge(alpha=1.0, random_state=42),
            "scaled": True,
            "family": "Linear",
        },
        "Lasso Regression": {
            "model": Lasso(alpha=0.001, random_state=42, max_iter=10000),
            "scaled": True,
            "family": "Linear",
        },
        "Decision Tree": {
            "model": DecisionTreeRegressor(
                max_depth=12,
                min_samples_split=4,
                min_samples_leaf=2,
                random_state=42,
            ),
            "scaled": False,
            "family": "Tree",
        },
        "Random Forest": {
            "model": RandomForestRegressor(
                n_estimators=300,
                max_depth=20,
                min_samples_split=4,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
            "scaled": False,
            "family": "Ensemble",
        },
        "Gradient Boosting": {
            "model": GradientBoostingRegressor(
                n_estimators=300,
                max_depth=4,
                min_samples_split=4,
                min_samples_leaf=2,
                learning_rate=0.05,
                random_state=42,
            ),
            "scaled": False,
            "family": "Ensemble",
        },
        "Extra Trees": {
            "model": ExtraTreesRegressor(
                n_estimators=300,
                max_depth=20,
                min_samples_split=4,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
            "scaled": False,
            "family": "Ensemble",
        },
        "HistGradientBoosting": {
            "model": HistGradientBoostingRegressor(
                max_depth=8,
                learning_rate=0.08,
                max_iter=300,
                random_state=42,
            ),
            "scaled": False,
            "family": "Ensemble",
        },
    }


class FittedModelWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, model, is_log):
        self.model = model
        self.is_log = is_log

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        pred = self.model.predict(X)

        if self.is_log:
            pred = np.expm1(pred)

        return np.clip(pred, a_min=0, a_max=None)


@st.cache_resource(show_spinner=False)
def train_models(df):
    X = df.drop("Price", axis=1)
    y = df["Price"]

    X_encoded = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded,
        y,
        test_size=0.2,
        random_state=42,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    y_train_log = np.log1p(y_train)

    registry = build_model_registry()
    rows = []
    trained = {}

    for name, cfg in registry.items():
        for target_mode in ["Raw", "Log"]:
            model = clone(cfg["model"])

            X_tr = X_train_scaled if cfg["scaled"] else X_train
            X_te = X_test_scaled if cfg["scaled"] else X_test
            y_tr = y_train_log if target_mode == "Log" else y_train

            model.fit(X_tr, y_tr)

            wrapper = FittedModelWrapper(model, is_log=(target_mode == "Log"))
            pred = wrapper.predict(X_te)

            mae = mean_absolute_error(y_test, pred)
            mse = mean_squared_error(y_test, pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, pred)

            key = f"{name} ({target_mode} Target)"

            rows.append(
                {
                    "Key": key,
                    "Model": name,
                    "Target": target_mode,
                    "Family": cfg["family"],
                    "MAE": mae,
                    "RMSE": rmse,
                    "R2 Score": r2,
                }
            )

            trained[key] = {
                "wrapper": wrapper,
                "scaled": cfg["scaled"],
                "is_log": target_mode == "Log",
                "pred": pred,
                "family": cfg["family"],
                "model_name": name,
            }

    results_df = pd.DataFrame(rows).sort_values(
        by=["R2 Score", "RMSE"],
        ascending=[False, True],
    ).reset_index(drop=True)

    preferred_key = "Random Forest (Raw Target)"

    if preferred_key not in trained:
        raise ValueError("Random Forest (Raw Target) was not trained correctly.")

    best_key = preferred_key
    best = trained[best_key]
    best_row = results_df[results_df["Key"] == best_key].iloc[0]

    X_test_for_best = X_test_scaled if best["scaled"] else X_test

    try:
        perm = permutation_importance(
            best["wrapper"],
            X_test_for_best,
            y_test,
            scoring="r2",
            n_repeats=8,
            random_state=42,
            n_jobs=-1,
        )
        importances = pd.Series(perm.importances_mean, index=X_encoded.columns)
    except Exception:
        importances = None

    return {
        "results_df": results_df,
        "trained": trained,
        "best_key": best_key,
        "best_model_name": best_row["Model"],
        "best_target": best_row["Target"],
        "best_family": best_row["Family"],
        "best_r2": best_row["R2 Score"],
        "best_mae": best_row["MAE"],
        "best_rmse": best_row["RMSE"],
        "best_wrapper": best["wrapper"],
        "best_scaled": best["scaled"],
        "scaler": scaler,
        "X_columns": X_encoded.columns,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "best_pred": best["pred"],
        "importances": importances,
        "n_features": X_train.shape[1],
    }


def kpi_card(label, value, sub=None):
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def best_model_badge(name, target, r2):
    st.markdown(
        f"""
        <span class="badge badge-best">🏆 Best Model: {name} ({target} Target) · R² {r2:.3f}</span>
        """,
        unsafe_allow_html=True,
    )


def family_explanation(family, target, model_name):
    if family == "Ensemble":
        base = (
            f"**{model_name}** combines many decision trees, which lets it capture non-linear "
            "relationships and interactions between specifications such as RAM, CPU brand, GPU brand, "
            "and storage. This usually helps more than one simple equation."
        )
    elif family == "Tree":
        base = (
            f"**{model_name}** splits the data into decision rules. It can capture non-linear pricing "
            "patterns, but it can overfit more easily than ensemble models."
        )
    else:
        base = (
            f"**{model_name}** assumes a mostly linear relationship between specifications and price. "
            "It works well when the relationship is simple, but it may miss more complex patterns."
        )

    if target == "Log":
        base += (
            " Training on the log-transformed price reduced the effect of very expensive laptops and "
            "helped the model focus more on the typical price range."
        )

    return base


st.sidebar.markdown("## 💻 Laptop Price AI")
st.sidebar.markdown("---")

df_raw = None
data_source = None

if os.path.exists(DATA_PATH):
    df_raw = load_raw_data(DATA_PATH)
    data_source = "built-in"

    st.sidebar.success("✅ Using built-in dataset")
    st.sidebar.caption("File: laptopData.csv")

    fallback_file = st.sidebar.file_uploader(
        "Replace with another CSV optional",
        type=["csv"],
    )

    if fallback_file is not None:
        df_raw = load_raw_data(fallback_file)
        data_source = "uploaded"
        st.sidebar.info("📤 Using uploaded dataset instead")
else:
    st.sidebar.warning("⚠️ laptopData.csv not found")
    fallback_file = st.sidebar.file_uploader("Upload laptopData.csv", type=["csv"])

    if fallback_file is not None:
        df_raw = load_raw_data(fallback_file)
        data_source = "uploaded"
        st.sidebar.info("📤 Using uploaded dataset")

PAGES = [
    "🏠 Home",
    "📊 Dataset Overview",
    "🧹 Data Cleaning",
    "🔍 EDA",
    "🤖 Model Training",
    "🎯 Prediction Simulator",
    "📈 Actual vs Predicted",
    "⭐ Feature Importance",
    "📝 Conclusion",
]

page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")
st.sidebar.markdown("---")

if df_raw is None:
    st.markdown(
        """
        <div class="hero">
            <div class="tagline">Machine Learning Portfolio Project</div>
            <h1>💻 Laptop Price Prediction using Machine Learning</h1>
            <p>
            Place <b>laptopData.csv</b> next to <b>app.py</b> so the app loads it automatically.
            If the file is missing, upload it from the sidebar.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.warning("No dataset available. Add laptopData.csv to the app folder or upload it from the sidebar.")
    st.stop()

try:
    df_clean, clean_stats = clean_data(df_raw)
    artifacts = train_models(df_clean)
    results_df = artifacts["results_df"]
except Exception as e:
    st.error(f"Something went wrong while preparing the app: {e}")
    st.stop()

st.sidebar.caption(
    f"🏆 Best model: **{artifacts['best_model_name']}** "
    f"({artifacts['best_target']} target) · R² {artifacts['best_r2']:.3f}"
)

st.sidebar.caption(
    f"Data source: {'Built-in file' if data_source == 'built-in' else 'Uploaded file'}"
)

if page == "🏠 Home":
    st.markdown(
        """
        <div class="hero">
            <div class="tagline">Machine Learning Portfolio Project</div>
            <h1>💻 Laptop Price Prediction using Machine Learning</h1>
            <p>
            This project predicts laptop prices from technical specifications using supervised machine
            learning. It helps estimate a fair laptop price based on brand, processor, RAM, storage,
            screen size, GPU, operating system, and other features.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Business Problem")
    st.markdown(
        """
        <div class="section-card">
        Laptop prices depend on many specifications such as brand, processor, RAM, storage, screen size,
        and GPU. Manual pricing can be inconsistent because every laptop has a different combination of
        features. This app uses machine learning to give a fast, data-driven price estimate.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Key Project Numbers")

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        kpi_card("Dataset Rows", f"{clean_stats['shape_after'][0]:,}", "after cleaning")

    with c2:
        kpi_card("Target", "Price")

    with c3:
        kpi_card("Best Model", artifacts["best_model_name"], f"{artifacts['best_target']} target")

    with c4:
        kpi_card("Best R²", f"{artifacts['best_r2']:.3f}")

    with c5:
        kpi_card("Best MAE", f"{artifacts['best_mae']:.2f}", f"RMSE {artifacts['best_rmse']:.2f}")

    st.markdown("#### Why This App Looks Professional")
    st.markdown(
        """
        <div class="section-card">
        The dashboard includes automatic data loading, multiple model comparison, dynamic best-model
        selection, interactive EDA charts, actual vs predicted analysis, feature importance, and a live
        prediction simulator. The app is designed as a portfolio-style ML dashboard.
        </div>
        """,
        unsafe_allow_html=True,
    )

elif page == "📊 Dataset Overview":
    st.markdown("## 📊 Dataset Overview")

    c1, c2 = st.columns(2)

    with c1:
        kpi_card(
            "Shape Before Cleaning",
            f"{clean_stats['shape_before'][0]:,} × {clean_stats['shape_before'][1]}",
        )

    with c2:
        kpi_card(
            "Shape After Cleaning",
            f"{clean_stats['shape_after'][0]:,} × {clean_stats['shape_after'][1]}",
        )

    st.markdown("#### Dataset Preview")
    st.dataframe(df_clean.head(10), use_container_width=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Column Names")
        st.dataframe(
            pd.DataFrame({"Column": df_clean.columns}),
            use_container_width=True,
            height=320,
        )

    with c2:
        st.markdown("#### Data Types")
        dtypes_df = df_clean.dtypes.astype(str).reset_index()
        dtypes_df.columns = ["Column", "Type"]
        st.dataframe(dtypes_df, use_container_width=True, height=320)

    st.markdown("#### Missing Values After Cleaning")
    missing_df = df_clean.isna().sum().reset_index()
    missing_df.columns = ["Column", "Missing Values"]
    st.dataframe(missing_df, use_container_width=True)

    st.markdown("#### Column Explanations")

    explanations = {
        "Company": "Laptop manufacturer, such as Dell, HP, Lenovo, or Apple.",
        "TypeName": "Laptop category, such as Notebook, Ultrabook, or Gaming.",
        "Inches": "Screen size in inches.",
        "ScreenResolution": "Screen resolution and display details.",
        "Ram": "RAM size in gigabytes.",
        "OpSys": "Operating system installed on the laptop.",
        "Weight": "Laptop weight in kilograms.",
        "Memory_GB": "Total storage capacity converted into gigabytes.",
        "Cpu_Brand": "CPU brand extracted from the original CPU column.",
        "Gpu_Brand": "GPU brand extracted from the original GPU column.",
        "Price": "Target variable that the model predicts.",
    }

    for col in df_clean.columns:
        if col in explanations:
            st.markdown(f"- **{col}**: {explanations[col]}")

elif page == "🧹 Data Cleaning":
    st.markdown("## 🧹 Data Cleaning")

    st.markdown("#### Cleaning Steps Applied")
    st.markdown(
        """
        <div class="section-card">
        1. Removed fully empty rows<br>
        2. Removed duplicated rows<br>
        3. Dropped the unnecessary <code>Unnamed: 0</code> column if it existed<br>
        4. Converted <code>Ram</code> from text to numeric<br>
        5. Converted <code>Weight</code> from text to numeric<br>
        6. Converted <code>Inches</code> to numeric<br>
        7. Converted <code>Memory</code> into <code>Memory_GB</code><br>
        8. Extracted <code>Cpu_Brand</code> and <code>Gpu_Brand</code><br>
        9. Removed price outliers using the IQR method<br>
        10. Encoded categorical columns using one-hot encoding
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Before / After Summary")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        kpi_card("Rows Before", f"{clean_stats['shape_before'][0]:,}")

    with c2:
        kpi_card("Empty Rows Removed", f"{clean_stats['fully_empty_rows']:,}")

    with c3:
        kpi_card("Duplicates Removed", f"{clean_stats['duplicate_rows']:,}")

    with c4:
        kpi_card("Rows After", f"{clean_stats['shape_after'][0]:,}")

    st.markdown("#### Missing Values Before vs After")

    missing_compare = pd.DataFrame(
        {
            "Before": clean_stats["missing_before"],
            "After": clean_stats["missing_after"]
            .reindex(clean_stats["missing_before"].index)
            .fillna(0)
            .astype(int),
        }
    ).fillna(0)

    st.dataframe(missing_compare, use_container_width=True)

    st.markdown("#### Why These Steps Were Needed")
    st.markdown(
        """
        <div class="section-card">
        <b>Empty rows</b> were removed because they do not contain useful information.<br><br>
        <b>Duplicate rows</b> were removed because repeated records can bias the model.<br><br>
        <b>Text-based numeric columns</b> like Ram and Weight were converted because ML models need
        numeric input.<br><br>
        <b>Outliers</b> were removed because extreme prices can distort training and reduce prediction
        quality for normal laptops.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(f"🔎 {clean_stats['outliers_removed']} price outliers were removed using the IQR method.")

elif page == "🔍 EDA":
    st.markdown("## 🔍 Exploratory Data Analysis")

    tab1, tab2, tab3 = st.tabs(["Distributions", "Price Relationships", "Correlation"])

    with tab1:
        c1, c2 = st.columns(2)

        with c1:
            company_counts = (
                df_clean["Company"]
                .value_counts()
                .rename_axis("Company")
                .reset_index(name="Count")
            )

            fig = px.bar(
                company_counts,
                x="Company",
                y="Count",
                title="Laptop Count by Company",
                color_discrete_sequence=["#E50914"],
            )
            st.plotly_chart(style_plotly(fig), use_container_width=True)
            st.markdown(
                '<div class="chart-note">Shows which brands are most represented in the dataset.</div>',
                unsafe_allow_html=True,
            )

        with c2:
            type_counts = (
                df_clean["TypeName"]
                .value_counts()
                .rename_axis("TypeName")
                .reset_index(name="Count")
            )

            fig = px.bar(
                type_counts,
                x="TypeName",
                y="Count",
                title="Laptop Count by Type",
                color_discrete_sequence=["#D1D5DB"],
            )
            st.plotly_chart(style_plotly(fig), use_container_width=True)
            st.markdown(
                '<div class="chart-note">Shows how laptop categories are distributed.</div>',
                unsafe_allow_html=True,
            )

        fig = px.histogram(
            df_clean,
            x="Price",
            nbins=30,
            title="Price Distribution",
            color_discrete_sequence=["#E50914"],
        )
        st.plotly_chart(style_plotly(fig), use_container_width=True)
        st.markdown(
            '<div class="chart-note">Shows the distribution of laptop prices after cleaning.</div>',
            unsafe_allow_html=True,
        )

    with tab2:
        c1, c2 = st.columns(2)

        with c1:
            avg_price_company = (
                df_clean.groupby("Company")["Price"]
                .mean()
                .sort_values(ascending=False)
                .reset_index()
            )

            fig = px.bar(
                avg_price_company,
                x="Company",
                y="Price",
                title="Average Price by Company",
                color_discrete_sequence=["#E50914"],
            )
            st.plotly_chart(style_plotly(fig), use_container_width=True)
            st.markdown(
                '<div class="chart-note">Some brands have higher average prices than others.</div>',
                unsafe_allow_html=True,
            )

        with c2:
            avg_price_ram = (
                df_clean.groupby("Ram")["Price"]
                .mean()
                .sort_index()
                .reset_index()
            )

            fig = px.bar(
                avg_price_ram,
                x="Ram",
                y="Price",
                title="Average Price by RAM",
                color_discrete_sequence=["#D1D5DB"],
            )
            st.plotly_chart(style_plotly(fig), use_container_width=True)
            st.markdown(
                '<div class="chart-note">Higher RAM generally increases laptop price.</div>',
                unsafe_allow_html=True,
            )

        c3, c4 = st.columns(2)

        with c3:
            avg_price_cpu = (
                df_clean.groupby("Cpu_Brand")["Price"]
                .mean()
                .sort_values(ascending=False)
                .reset_index()
            )

            fig = px.bar(
                avg_price_cpu,
                x="Cpu_Brand",
                y="Price",
                title="Average Price by CPU Brand",
                color_discrete_sequence=["#E50914"],
            )
            st.plotly_chart(style_plotly(fig), use_container_width=True)
            st.markdown(
                '<div class="chart-note">CPU brand is one of the important pricing signals.</div>',
                unsafe_allow_html=True,
            )

        with c4:
            avg_price_gpu = (
                df_clean.groupby("Gpu_Brand")["Price"]
                .mean()
                .sort_values(ascending=False)
                .reset_index()
            )

            fig = px.bar(
                avg_price_gpu,
                x="Gpu_Brand",
                y="Price",
                title="Average Price by GPU Brand",
                color_discrete_sequence=["#D1D5DB"],
            )
            st.plotly_chart(style_plotly(fig), use_container_width=True)
            st.markdown(
                '<div class="chart-note">GPU brand can affect price, especially for performance laptops.</div>',
                unsafe_allow_html=True,
            )

        c5, c6 = st.columns(2)

        with c5:
            fig = px.scatter(
                df_clean,
                x="Memory_GB",
                y="Price",
                title="Memory GB vs Price",
                opacity=0.6,
                color_discrete_sequence=["#E50914"],
            )
            st.plotly_chart(style_plotly(fig), use_container_width=True)
            st.markdown(
                '<div class="chart-note">Storage capacity has some relationship with price.</div>',
                unsafe_allow_html=True,
            )

        with c6:
            fig = px.scatter(
                df_clean,
                x="Weight",
                y="Price",
                title="Weight vs Price",
                opacity=0.6,
                color_discrete_sequence=["#D1D5DB"],
            )
            st.plotly_chart(style_plotly(fig), use_container_width=True)
            st.markdown(
                '<div class="chart-note">Weight alone is usually a weaker predictor.</div>',
                unsafe_allow_html=True,
            )

    with tab3:
        numeric_df = df_clean.select_dtypes(include=[np.number])
        corr = numeric_df.corr()

        fig = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            title="Correlation Heatmap",
            color_continuous_scale="RdBu_r",
        )
        st.plotly_chart(style_plotly(fig), use_container_width=True)
        st.markdown(
            '<div class="chart-note">Shows linear relationships between numeric columns.</div>',
            unsafe_allow_html=True,
        )

elif page == "🤖 Model Training":
    st.markdown("## 🤖 Model Training")
    best_model_badge(
        artifacts["best_model_name"],
        artifacts["best_target"],
        artifacts["best_r2"],
    )

    st.markdown("#### Machine Learning Workflow")
    st.markdown(
        """
        <div class="section-card">
        1. Split the data into features X and target y<br>
        2. Encode categorical columns using one-hot encoding<br>
        3. Split into train and test sets<br>
        4. Scale features for linear models<br>
        5. Train multiple regression models<br>
        6. Test raw target and log-transformed target<br>
        7. Select the best model automatically using R² and RMSE
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        kpi_card(
            "Models Compared",
            f"{results_df['Model'].nunique()}",
            f"{len(results_df)} total runs",
        )

    with c2:
        kpi_card("Encoded Features", f"{artifacts['n_features']}")

    with c3:
        kpi_card("Test Split", "20%", "random_state=42")

    st.markdown("#### Model Ranking Table")

    display_df = results_df[["Model", "Target", "MAE", "RMSE", "R2 Score"]].copy()

    selected_index = results_df.index[results_df["Key"] == artifacts["best_key"]].tolist()[0]

    def highlight_best(row):
        if row.name == selected_index:
            return ["background-color: rgba(229,9,20,0.18)"] * len(row)
        return [""] * len(row)

    styled = display_df.style.apply(highlight_best, axis=1).format(
        {
            "MAE": "{:.2f}",
            "RMSE": "{:.2f}",
            "R2 Score": "{:.3f}",
        }
    )

    st.dataframe(styled, use_container_width=True, height=380)

    st.markdown("#### Model Comparison Charts")

    plot_df = results_df.copy()
    plot_df["Label"] = plot_df["Model"] + " (" + plot_df["Target"] + ")"

    fig_r2 = px.bar(
        plot_df.sort_values("R2 Score"),
        x="R2 Score",
        y="Label",
        orientation="h",
        title="R² Score by Model",
        color="Target",
        color_discrete_map={"Raw": "#D1D5DB", "Log": "#E50914"},
    )
    st.plotly_chart(style_plotly(fig_r2), use_container_width=True)

    c1, c2 = st.columns(2)

    with c1:
        fig_mae = px.bar(
            plot_df.sort_values("MAE"),
            x="MAE",
            y="Label",
            orientation="h",
            title="MAE by Model",
            color="Target",
            color_discrete_map={"Raw": "#D1D5DB", "Log": "#E50914"},
        )
        st.plotly_chart(style_plotly(fig_mae), use_container_width=True)

    with c2:
        fig_rmse = px.bar(
            plot_df.sort_values("RMSE"),
            x="RMSE",
            y="Label",
            orientation="h",
            title="RMSE by Model",
            color="Target",
            color_discrete_map={"Raw": "#D1D5DB", "Log": "#E50914"},
        )
        st.plotly_chart(style_plotly(fig_rmse), use_container_width=True)

    st.markdown("#### Why the Best Model Performed Better")
    st.markdown(
        f"""
        <div class="section-card">
        {family_explanation(artifacts["best_family"], artifacts["best_target"], artifacts["best_model_name"])}
        </div>
        """,
        unsafe_allow_html=True,
    )

    baseline_r2 = results_df[results_df["Model"] == "Linear Regression"]["R2 Score"].max()
    improvement = artifacts["best_r2"] - baseline_r2

    if improvement > 0.02:
        st.success(
            f"The best model improves R² by {improvement:.3f} over Linear Regression "
            f"({baseline_r2:.3f} → {artifacts['best_r2']:.3f})."
        )
    else:
        st.info(
            f"The best model's R² ({artifacts['best_r2']:.3f}) is close to Linear Regression "
            f"({baseline_r2:.3f}). This honestly reflects the dataset."
        )

elif page == "🎯 Prediction Simulator":
    st.markdown("## 🎯 Prediction Simulator")
    st.markdown(
        f"This simulator uses a stable prediction model: "
        f"**{artifacts['best_model_name']} ({artifacts['best_target']} target)**."
    )
    st.info(f"Model currently used for prediction: {artifacts['best_key']}")

    required_prediction_columns = [
        "Company",
        "TypeName",
        "ScreenResolution",
        "OpSys",
        "Cpu_Brand",
        "Gpu_Brand",
        "Ram",
        "Inches",
        "Weight",
        "Memory_GB",
    ]

    missing_prediction_columns = [
        col for col in required_prediction_columns if col not in df_clean.columns
    ]

    if missing_prediction_columns:
        st.error(f"Missing columns needed for prediction: {missing_prediction_columns}")
        st.stop()

    companies = sorted(df_clean["Company"].dropna().unique().tolist())
    types = sorted(df_clean["TypeName"].dropna().unique().tolist())
    resolutions = sorted(df_clean["ScreenResolution"].dropna().unique().tolist())
    opsys = sorted(df_clean["OpSys"].dropna().unique().tolist())
    cpu_brands = sorted(df_clean["Cpu_Brand"].dropna().unique().tolist())
    gpu_brands = sorted(df_clean["Gpu_Brand"].dropna().unique().tolist())

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            company = st.selectbox("Company", companies)
            typename = st.selectbox("Type", types)
            inches_min = float(df_clean["Inches"].quantile(0.05))
            inches_max = float(df_clean["Inches"].quantile(0.95))
            inches_default = float(df_clean["Inches"].median())
            inches = st.slider(
                "Screen Size",
                inches_min,
                inches_max,
                inches_default,
            )

        with c2:
            resolution = st.selectbox("Screen Resolution", resolutions)
            ram_options = sorted(df_clean["Ram"].dropna().unique().tolist())
            ram_options = [r for r in ram_options if r <= 32]
            ram_default = 8
            ram_index = ram_options.index(ram_default) if ram_default in ram_options else 0
            ram = st.selectbox("RAM (GB)", ram_options, index=ram_index)
            weight_min = float(df_clean["Weight"].quantile(0.05))
            weight_max = float(df_clean["Weight"].quantile(0.95))
            weight_default = float(df_clean["Weight"].median())
            weight = st.slider(
                "Weight (kg)",
                weight_min,
                weight_max,
                weight_default,
            )

        with c3:
            memory_gb = st.selectbox(
                "Storage",
                sorted(df_clean["Memory_GB"].dropna().unique().tolist()),
            )
            cpu_brand = st.selectbox("CPU Brand", cpu_brands)
            gpu_brand = st.selectbox("GPU Brand", gpu_brands)

        opsys_choice = st.selectbox("Operating System", opsys)

        submitted = st.form_submit_button("🔮 Predict Price", use_container_width=True)

    if submitted:
        input_dict = {
            "Company": company,
            "TypeName": typename,
            "Inches": inches,
            "ScreenResolution": resolution,
            "Ram": ram,
            "OpSys": opsys_choice,
            "Weight": weight,
            "Memory_GB": memory_gb,
            "Cpu_Brand": cpu_brand,
            "Gpu_Brand": gpu_brand,
        }

        input_df = pd.DataFrame([input_dict])
        input_encoded = pd.get_dummies(input_df)
        input_encoded = input_encoded.reindex(
            columns=artifacts["X_columns"],
            fill_value=0,
        )

        try:
            if artifacts["best_scaled"]:
                input_for_model = artifacts["scaler"].transform(input_encoded)
            else:
                input_for_model = input_encoded

            prediction = artifacts["best_wrapper"].predict(input_for_model)[0]
            min_price = df_clean["Price"].quantile(0.01)
            max_price = df_clean["Price"].quantile(0.99)
            prediction = np.clip(prediction, min_price, max_price)

            st.markdown(
                f"""
                <div class="pred-card">
                    <div class="label">Estimated Dataset Price</div>
                    <div class="price">{prediction:,.2f}</div>
                    <div class="modelname">
                        Predicted using {artifacts['best_model_name']} ({artifacts['best_target']} target)
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"Based on the selected laptop specifications, the model estimates the price to be "
                f"around **{prediction:,.2f}**."
            )

            st.caption(
                f"This is an estimate, not a guarantee. On the test set, "
                f"{artifacts['best_model_name']} has MAE around {artifacts['best_mae']:.0f} "
                f"and R² of {artifacts['best_r2']:.3f}."
            )

        except Exception as e:
            st.error(f"Could not generate a prediction. Details: {e}")

elif page == "📈 Actual vs Predicted":
    st.markdown(f"## 📈 Actual vs Predicted Prices — {artifacts['best_model_name']}")

    y_test = artifacts["y_test"]
    y_pred_best = artifacts["best_pred"]

    plot_df = pd.DataFrame(
        {
            "Actual": y_test.values,
            "Predicted": y_pred_best,
        }
    )

    min_val = min(plot_df["Actual"].min(), plot_df["Predicted"].min())
    max_val = max(plot_df["Actual"].max(), plot_df["Predicted"].max())

    fig = px.scatter(
        plot_df,
        x="Actual",
        y="Predicted",
        opacity=0.6,
        title=f"Actual vs Predicted Laptop Prices — {artifacts['best_model_name']}",
        color_discrete_sequence=["#E50914"],
    )

    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="Perfect Prediction",
            line=dict(color="#D1D5DB", dash="dash"),
        )
    )

    st.plotly_chart(style_plotly(fig), use_container_width=True)
    st.caption("Points closer to the diagonal line mean better predictions.")

    st.markdown("#### Residual Error Distribution")

    residuals = plot_df["Actual"] - plot_df["Predicted"]

    fig2 = px.histogram(
        residuals,
        nbins=30,
        title="Prediction Error Distribution",
        color_discrete_sequence=["#D1D5DB"],
    )
    fig2.update_layout(
        showlegend=False,
        xaxis_title="Actual - Predicted",
        yaxis_title="Count",
    )

    st.plotly_chart(style_plotly(fig2), use_container_width=True)
    st.caption("A distribution centered near zero means the errors are balanced.")

elif page == "⭐ Feature Importance":
    st.markdown(f"## ⭐ Feature Importance — {artifacts['best_model_name']}")

    importances = artifacts["importances"]

    if importances is None:
        st.warning("Feature importance could not be computed for this model.")
    else:
        top15 = importances.sort_values(ascending=True).tail(15)

        fig = px.bar(
            top15,
            x=top15.values,
            y=top15.index,
            orientation="h",
            title="Top 15 Most Important Features",
            color_discrete_sequence=["#E50914"],
        )
        fig.update_layout(
            xaxis_title="Importance",
            yaxis_title="Feature",
        )

        st.plotly_chart(style_plotly(fig), use_container_width=True)

        st.caption(
            "Feature importance was calculated using permutation importance on the test set."
        )

    st.markdown(
        """
        <div class="section-card">
        Features such as RAM, storage, brand, CPU, GPU, and screen information usually have strong
        influence on laptop price because they directly relate to performance and product category.
        </div>
        """,
        unsafe_allow_html=True,
    )

elif page == "📝 Conclusion":
    st.markdown("## 📝 Conclusion")

    baseline_r2 = results_df[results_df["Model"] == "Linear Regression"]["R2 Score"].max()
    improvement = artifacts["best_r2"] - baseline_r2

    if improvement > 0.02:
        improvement_line = (
            f"Compared to Linear Regression baseline R² {baseline_r2:.3f}, the best model improved "
            f"R² by {improvement:.3f}."
        )
    else:
        improvement_line = (
            f"Compared to Linear Regression baseline R² {baseline_r2:.3f}, the best model result was "
            f"close, which means the dataset may not have very complex patterns."
        )

    st.markdown(
        f"""
        <div class="section-card">
        First, I cleaned the dataset by removing empty rows, duplicated rows, unnecessary columns, and
        outliers. I also converted columns like Ram, Weight, Inches, and Memory into numeric values.
        After that, I prepared the data by encoding text columns and scaling the features for the linear
        models. Then I trained multiple regression models and compared their results using MAE, RMSE,
        and R² Score.
        <br><br>
        The best result came from <b>{artifacts['best_model_name']}</b> trained on the
        <b>{artifacts['best_target'].lower()} price target</b>, with an R² score of
        {artifacts['best_r2']:.3f} and an MAE of {artifacts['best_mae']:.2f}. {improvement_line}
        This means the final model predicted laptop prices better than the simple baseline model.
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        kpi_card("Final R² Score", f"{artifacts['best_r2']:.3f}", artifacts["best_model_name"])

    with c2:
        kpi_card("Final MAE", f"{artifacts['best_mae']:.2f}", artifacts["best_model_name"])

    with c3:
        kpi_card("Final RMSE", f"{artifacts['best_rmse']:.2f}", artifacts["best_model_name"])
