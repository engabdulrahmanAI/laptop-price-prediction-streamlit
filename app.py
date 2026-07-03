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

# Make all Plotly charts readable on the dark Streamlit theme.
pio.templates.default = "plotly_dark"


def style_plotly(fig):
    fig.update_layout(
        paper_bgcolor="#0B0B0F",
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

    html, body, [class*="css"] {
        font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
    }

    .main {
        background-color: #0B0B0F;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        background: linear-gradient(135deg, #16161D 0%, #1F1F2A 60%, #2A0A0E 100%);
        padding: 2.8rem 2.6rem;
        border-radius: 16px;
        margin-bottom: 1.6rem;
        border: 1px solid #2A2A35;
    }
    .hero h1 {
        color: #FFFFFF;
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .hero p {
        color: #D1D5DB;
        font-size: 1.05rem;
        max-width: 800px;
        line-height: 1.6;
    }
    .hero .tagline {
        display: inline-block;
        color: #E50914;
        font-weight: 700;
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
    }
    .kpi-label {
        color: #D1D5DB;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.35rem;
    }
    .kpi-value {
        color: #FFFFFF;
        font-size: 1.75rem;
        font-weight: 800;
    }
    .kpi-sub {
        color: #E50914;
        font-size: 0.8rem;
        margin-top: 0.25rem;
        font-weight: 600;
    }

    .section-card {
        background: #16161D;
        border: 1px solid #2A2A35;
        border-radius: 12px;
        padding: 1.5rem 1.6rem;
        margin-bottom: 1rem;
        color: #D1D5DB;
        line-height: 1.6;
    }
    .section-card b {
        color: #FFFFFF;
    }

    .chart-note {
        color: #D1D5DB;
        font-size: 0.88rem;
        margin-top: -0.4rem;
        margin-bottom: 1rem;
        padding-left: 0.1rem;
    }

    .pred-card {
        background: linear-gradient(135deg, #1F1F2A 0%, #2A0A0E 100%);
        border-radius: 16px;
        padding: 2.2rem;
        text-align: center;
        border: 1px solid #E50914;
    }
    .pred-card .price {
        font-size: 2.8rem;
        font-weight: 800;
        color: #FFFFFF;
    }
    .pred-card .label {
        color: #D1D5DB;
        font-size: 0.9rem;
        margin-bottom: 0.6rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .pred-card .modelname {
        color: #E50914;
        font-weight: 700;
        font-size: 0.95rem;
        margin-top: 0.6rem;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.6rem;
        color: #FFFFFF;
    }

    .badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        margin-right: 0.4rem;
    }
    .badge-best {
        background: rgba(229, 9, 20, 0.15);
        color: #E50914;
        border: 1px solid #E50914;
    }

    section[data-testid="stSidebar"] {
        background-color: #0B0B0F;
        border-right: 1px solid #2A2A35;
    }

    .stButton > button, .stFormSubmitButton > button {
        background-color: #E50914;
        color: #FFFFFF;
        border: none;
        border-radius: 8px;
        font-weight: 700;
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover {
        background-color: #B20710;
        color: #FFFFFF;
    }

    /* Improve text readability across Streamlit widgets */
    h1, h2, h3, h4, h5, h6, p, span, label, div {
        color: #FFFFFF;
    }

    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stMarkdownContainer"] li {
        color: #D1D5DB;
    }

    .stSelectbox label, .stSlider label, .stNumberInput label, .stTextInput label,
    .stFileUploader label, .stRadio label {
        color: #FFFFFF !important;
        font-weight: 600;
    }

    div[data-baseweb="select"] > div,
    div[data-testid="stNumberInput"] input,
    div[data-testid="stTextInput"] input {
        background-color: #16161D !important;
        color: #FFFFFF !important;
        border-color: #2A2A35 !important;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #2A2A35;
        border-radius: 10px;
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
        if "TB" in size:
            number = float(size.replace("TB", ""))
            total += number * 1024
        elif "GB" in size:
            number = float(size.replace("GB", ""))
            total += number
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
        "Linear Regression": {"model": LinearRegression(), "scaled": True, "family": "Linear"},
        "Ridge Regression": {"model": Ridge(alpha=1.0, random_state=42), "scaled": True, "family": "Linear"},
        "Lasso Regression": {"model": Lasso(alpha=0.001, random_state=42, max_iter=10000), "scaled": True, "family": "Linear"},
        "Decision Tree": {
            "model": DecisionTreeRegressor(max_depth=12, min_samples_split=4, min_samples_leaf=2, random_state=42),
            "scaled": False, "family": "Tree",
        },
        "Random Forest": {
            "model": RandomForestRegressor(
                n_estimators=300, max_depth=20, min_samples_split=4, min_samples_leaf=2,
                random_state=42, n_jobs=-1,
            ),
            "scaled": False, "family": "Ensemble",
        },
        "Gradient Boosting": {
            "model": GradientBoostingRegressor(
                n_estimators=300, max_depth=4, min_samples_split=4, min_samples_leaf=2,
                learning_rate=0.05, random_state=42,
            ),
            "scaled": False, "family": "Ensemble",
        },
        "Extra Trees": {
            "model": ExtraTreesRegressor(
                n_estimators=300, max_depth=20, min_samples_split=4, min_samples_leaf=2,
                random_state=42, n_jobs=-1,
            ),
            "scaled": False, "family": "Ensemble",
        },
        "HistGradientBoosting": {
            "model": HistGradientBoostingRegressor(max_depth=8, learning_rate=0.08, max_iter=300, random_state=42),
            "scaled": False, "family": "Ensemble",
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
        X_encoded, y, test_size=0.2, random_state=42
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
            rows.append({
                "Key": key, "Model": name, "Target": target_mode, "Family": cfg["family"],
                "MAE": mae, "RMSE": rmse, "R2 Score": r2,
            })
            trained[key] = {
                "wrapper": wrapper, "scaled": cfg["scaled"], "is_log": target_mode == "Log",
                "pred": pred, "family": cfg["family"], "model_name": name,
            }

    results_df = pd.DataFrame(rows).sort_values(
        by=["R2 Score", "RMSE"], ascending=[False, True]
    ).reset_index(drop=True)

    best_key = results_df.iloc[0]["Key"]
    best = trained[best_key]

    X_test_for_best = X_test_scaled if best["scaled"] else X_test
    try:
        perm = permutation_importance(
            best["wrapper"], X_test_for_best, y_test,
            scoring="r2", n_repeats=8, random_state=42, n_jobs=-1,
        )
        importances = pd.Series(perm.importances_mean, index=X_encoded.columns)
    except Exception:
        importances = None

    return {
        "results_df": results_df,
        "trained": trained,
        "best_key": best_key,
        "best_model_name": results_df.iloc[0]["Model"],
        "best_target": results_df.iloc[0]["Target"],
        "best_family": results_df.iloc[0]["Family"],
        "best_r2": results_df.iloc[0]["R2 Score"],
        "best_mae": results_df.iloc[0]["MAE"],
        "best_rmse": results_df.iloc[0]["RMSE"],
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
            "relationships and interactions between specifications (e.g. how RAM and CPU brand "
            "jointly affect price) that a single linear equation cannot represent."
        )
    elif family == "Tree":
        base = (
            f"**{model_name}** splits the data into decision rules, which can capture non-linear "
            "pricing patterns but is more prone to overfitting than an ensemble of trees."
        )
    else:
        base = (
            f"**{model_name}** assumes a mostly linear relationship between specifications and price. "
            "It performed competitively, suggesting the dataset has a fairly linear pricing structure "
            "for this configuration."
        )
    if target == "Log":
        base += (
            " Training on the **log-transformed price** reduced the impact of a few very expensive "
            "laptops on the loss function, which helped the model fit the majority of typical-priced "
            "laptops more accurately."
        )
    return base


st.sidebar.markdown("## 💻 Laptop Price AI")
st.sidebar.markdown("---")

df_raw = None
data_source = None

if os.path.exists(DATA_PATH):
    df_raw = load_raw_data(DATA_PATH)
    data_source = "built-in"
    st.sidebar.success("✅ Using built-in dataset (laptopData.csv)")
    fallback_file = st.sidebar.file_uploader("Replace with a different laptopData.csv (optional)", type=["csv"])
    if fallback_file is not None:
        df_raw = load_raw_data(fallback_file)
        data_source = "uploaded"
        st.sidebar.info("📤 Using uploaded dataset instead of the built-in file")
else:
    st.sidebar.warning("⚠️ Built-in laptopData.csv not found in the app folder.")
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
            <p>Place <b>laptopData.csv</b> in the app folder so it loads automatically, or upload it
            from the sidebar, to explore the dataset, review the cleaning pipeline, compare models,
            and predict laptop prices.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.warning("⚠️ No dataset available yet. Add **laptopData.csv** to the app folder or upload it from the sidebar.")
    st.stop()

df_clean, clean_stats = clean_data(df_raw)
artifacts = train_models(df_clean)
results_df = artifacts["results_df"]

st.sidebar.caption(
    f"🏆 Best model: **{artifacts['best_model_name']}** ({artifacts['best_target']} target) · "
    f"R² {artifacts['best_r2']:.3f}"
)
st.sidebar.caption(f"Data source: {'Built-in file' if data_source == 'built-in' else 'Uploaded file'}")

if page == "🏠 Home":
    st.markdown(
        """
        <div class="hero">
            <div class="tagline">Machine Learning Portfolio Project</div>
            <h1>💻 Laptop Price Prediction using Machine Learning</h1>
            <p>This project predicts laptop prices from their technical specifications using supervised
            machine learning. It supports buyers, sellers, and retailers who need a fast, data-driven
            estimate of a fair laptop price instead of relying on guesswork. The dataset loads
            automatically from <b>laptopData.csv</b> bundled with the app.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Business Problem")
    st.markdown(
        """
        <div class="section-card">
        Laptop prices depend on many specifications — brand, processor, RAM, storage, screen size, and
        GPU — which makes manual pricing inconsistent. Retailers can under-price or over-price laptops,
        and buyers struggle to know whether a listed price is fair. A machine learning model that learns
        the relationship between specifications and price can produce a consistent, data-driven price
        estimate in seconds.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Key Project Numbers")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        kpi_card("Dataset Rows (Cleaned)", f"{clean_stats['shape_after'][0]:,}")
    with c2:
        kpi_card("Target Variable", "Price")
    with c3:
        kpi_card("Best Model", artifacts["best_model_name"], sub=f"{artifacts['best_target']} target")
    with c4:
        kpi_card("Best R² Score", f"{artifacts['best_r2']:.3f}")
    with c5:
        kpi_card("Best MAE", f"{artifacts['best_mae']:.2f}", sub=f"RMSE {artifacts['best_rmse']:.2f}")

    st.markdown("#### Why This Matters")
    st.markdown(
        """
        <div class="section-card">
        A reliable price prediction model helps e-commerce platforms auto-suggest fair prices for new
        listings, helps buyers spot overpriced or underpriced laptops, and gives retailers a benchmark
        for pricing strategy across brands and configurations.
        </div>
        """,
        unsafe_allow_html=True,
    )

elif page == "📊 Dataset Overview":
    st.markdown("## 📊 Dataset Overview")

    c1, c2 = st.columns(2)
    with c1:
        kpi_card("Shape Before Cleaning", f"{clean_stats['shape_before'][0]:,} × {clean_stats['shape_before'][1]}")
    with c2:
        kpi_card("Shape After Cleaning", f"{clean_stats['shape_after'][0]:,} × {clean_stats['shape_after'][1]}")

    st.markdown("#### Dataset Preview")
    st.dataframe(df_clean.head(10), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Column Names")
        st.dataframe(pd.DataFrame({"Column": df_clean.columns}), use_container_width=True, height=320)
    with c2:
        st.markdown("#### Data Types")
        st.dataframe(df_clean.dtypes.astype(str).reset_index().rename(
            columns={"index": "Column", 0: "Type"}), use_container_width=True, height=320)

    st.markdown("#### Missing Values (Cleaned Dataset)")
    st.dataframe(df_clean.isna().sum().reset_index().rename(
        columns={"index": "Column", 0: "Missing Values"}), use_container_width=True)

    st.markdown("#### Column Explanations")
    explanations = {
        "Company": "The laptop manufacturer, e.g. Dell, HP, Apple.",
        "TypeName": "The laptop category, e.g. Notebook, Ultrabook, Gaming.",
        "Inches": "Screen size in inches.",
        "ScreenResolution": "Screen resolution and panel details.",
        "Ram": "RAM size in gigabytes (GB).",
        "OpSys": "Operating system installed on the laptop.",
        "Weight": "Laptop weight in kilograms (kg).",
        "Memory_GB": "Total storage capacity in gigabytes, combining HDD/SSD/hybrid values.",
        "Cpu_Brand": "Processor brand/family extracted from the original CPU text.",
        "Gpu_Brand": "Graphics card brand extracted from the original GPU text.",
        "Price": "Target variable — the laptop price to predict.",
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
        3. Dropped the unnecessary <code>Unnamed: 0</code> index column<br>
        4. Converted <code>Ram</code> from text (e.g. "8GB") to numeric<br>
        5. Converted <code>Weight</code> from text (e.g. "1.5kg") to numeric<br>
        6. Converted <code>Inches</code> to numeric<br>
        7. Parsed <code>Memory</code> into a new numeric column <code>Memory_GB</code><br>
        8. Extracted <code>Cpu_Brand</code> and <code>Gpu_Brand</code> from the CPU/GPU text<br>
        9. Removed price outliers using the IQR method<br>
        10. Encoded categorical columns using one-hot encoding<br>
        11. Scaled features for linear-family models (Linear/Ridge/Lasso Regression)
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Before / After Summary")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Rows Before", f"{clean_stats['shape_before'][0]:,}")
    with c2:
        kpi_card("Fully Empty Rows Removed", f"{clean_stats['fully_empty_rows']:,}")
    with c3:
        kpi_card("Duplicate Rows Removed", f"{clean_stats['duplicate_rows']:,}")
    with c4:
        kpi_card("Rows After", f"{clean_stats['shape_after'][0]:,}")

    st.markdown("#### Missing Values Before vs After")
    missing_compare = pd.DataFrame({
        "Before": clean_stats["missing_before"],
        "After": clean_stats["missing_after"].reindex(clean_stats["missing_before"].index).fillna(0).astype(int),
    }).fillna(0)
    st.dataframe(missing_compare, use_container_width=True)

    st.markdown("#### Why These Steps Were Needed")
    st.markdown(
        """
        <div class="section-card">
        <b>Empty rows</b> carry no information and would only add noise, so they were removed.<br><br>
        <b>Duplicate rows</b> can bias the model toward repeated records, so they were dropped.<br><br>
        <b>Ram, Weight, Inches, and Memory</b> were stored as text (e.g. "8GB", "1.5kg"), which cannot be
        used directly by a numeric model, so each was converted into a proper numeric value.<br><br>
        <b>Outliers</b> in Price were removed using the IQR method because extremely high or unusual
        prices can distort the model and make it less accurate for typical laptops.
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
            company_counts = df_clean["Company"].value_counts().rename_axis("Company").reset_index(name="Count")
            fig = px.bar(
                company_counts,
                x="Company", y="Count", title="Laptop Count by Company",
                color_discrete_sequence=["#E50914"],
            )
            st.plotly_chart(style_plotly(fig), use_container_width=True)
            st.markdown('<div class="chart-note">Shows which brands are most represented in the dataset.</div>', unsafe_allow_html=True)
        with c2:
            type_counts = df_clean["TypeName"].value_counts().rename_axis("TypeName").reset_index(name="Count")
            fig = px.bar(
                type_counts,
                x="TypeName", y="Count", title="Laptop Count by Type",
                color_discrete_sequence=["#D1D5DB"],
            )
            st.plotly_chart(style_plotly(fig), use_container_width=True)
            st.markdown('<div class="chart-note">Shows how laptop categories (Notebook, Gaming, etc.) are distributed.</div>', unsafe_allow_html=True)

        fig = px.histogram(df_clean, x="Price", nbins=30, title="Price Distribution",
                            color_discrete_sequence=["#E50914"])
        st.plotly_chart(style_plotly(fig), use_container_width=True)
        st.markdown('<div class="chart-note">Most laptops fall in the lower-to-mid price range after outlier removal.</div>', unsafe_allow_html=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            avg_price_company = df_clean.groupby("Company")["Price"].mean().sort_values(ascending=False).reset_index()
            fig = px.bar(avg_price_company, x="Company", y="Price", title="Average Price by Company",
                         color_discrete_sequence=["#E50914"])
            st.plotly_chart(style_plotly(fig), use_container_width=True)
            st.markdown('<div class="chart-note">Brand alone shifts the average price noticeably.</div>', unsafe_allow_html=True)
        with c2:
            avg_price_ram = df_clean.groupby("Ram")["Price"].mean().sort_index().reset_index()
            fig = px.bar(avg_price_ram, x="Ram", y="Price", title="Average Price by RAM (GB)",
                         color_discrete_sequence=["#D1D5DB"])
            st.plotly_chart(style_plotly(fig), use_container_width=True)
            st.markdown('<div class="chart-note">Higher RAM configurations trend toward higher prices.</div>', unsafe_allow_html=True)

        c3, c4 = st.columns(2)
        with c3:
            avg_price_cpu = df_clean.groupby("Cpu_Brand")["Price"].mean().sort_values(ascending=False).reset_index()
            fig = px.bar(avg_price_cpu, x="Cpu_Brand", y="Price", title="Average Price by CPU Brand",
                         color_discrete_sequence=["#E50914"])
            st.plotly_chart(style_plotly(fig), use_container_width=True)
            st.markdown('<div class="chart-note">CPU brand/tier is a strong price signal.</div>', unsafe_allow_html=True)
        with c4:
            avg_price_gpu = df_clean.groupby("Gpu_Brand")["Price"].mean().sort_values(ascending=False).reset_index()
            fig = px.bar(avg_price_gpu, x="Gpu_Brand", y="Price", title="Average Price by GPU Brand",
                         color_discrete_sequence=["#D1D5DB"])
            st.plotly_chart(style_plotly(fig), use_container_width=True)
            st.markdown('<div class="chart-note">Dedicated GPU brands typically command higher prices.</div>', unsafe_allow_html=True)

        c5, c6 = st.columns(2)
        with c5:
            fig = px.scatter(df_clean, x="Memory_GB", y="Price", title="Memory (GB) vs Price",
                              opacity=0.6, color_discrete_sequence=["#E50914"])
            st.plotly_chart(style_plotly(fig), use_container_width=True)
            st.markdown('<div class="chart-note">More total storage is loosely associated with higher prices.</div>', unsafe_allow_html=True)
        with c6:
            fig = px.scatter(df_clean, x="Weight", y="Price", title="Weight vs Price",
                              opacity=0.6, color_discrete_sequence=["#D1D5DB"])
            st.plotly_chart(style_plotly(fig), use_container_width=True)
            st.markdown('<div class="chart-note">Weight alone is a weaker predictor than RAM or brand.</div>', unsafe_allow_html=True)

    with tab3:
        numeric_df = df_clean.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        fig = px.imshow(corr, text_auto=".2f", aspect="auto", title="Correlation Heatmap (Numeric Columns)",
                         color_continuous_scale="RdBu_r")
        st.plotly_chart(style_plotly(fig), use_container_width=True)
        st.markdown('<div class="chart-note">Darker red/blue cells indicate stronger linear relationships between numeric columns.</div>', unsafe_allow_html=True)

elif page == "🤖 Model Training":
    st.markdown("## 🤖 Model Training")
    best_model_badge(artifacts["best_model_name"], artifacts["best_target"], artifacts["best_r2"])
    st.markdown("")

    st.markdown("#### Machine Learning Workflow")
    st.markdown(
        """
        <div class="section-card">
        1. Split the cleaned data into features (X) and target (y = Price)<br>
        2. One-hot encode categorical columns<br>
        3. Split into training and test sets (80/20, <code>random_state=42</code>)<br>
        4. Scale features with StandardScaler for linear-family models (Linear, Ridge, Lasso); tree-based
        models train on the unscaled encoded features<br>
        5. Train each model twice: once on the raw <code>Price</code>, once on <code>log1p(Price)</code>
        with predictions converted back using <code>expm1</code><br>
        6. Evaluate every model/target combination on the same test set using MAE, RMSE, and R² Score<br>
        7. Automatically select the best model by highest R² Score, breaking ties with lowest RMSE
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card("Models Compared", f"{results_df['Model'].nunique()}", sub=f"{len(results_df)} total runs (raw + log)")
    with c2:
        kpi_card("Encoded Features", f"{artifacts['n_features']}")
    with c3:
        kpi_card("Test Split", "20%", sub="random_state=42")

    st.markdown("#### Model Ranking Table")
    display_df = results_df[["Model", "Target", "MAE", "RMSE", "R2 Score"]].copy()

    def highlight_best(row):
        if row.name == 0:
            return ["background-color: rgba(229,9,20,0.18)"] * len(row)
        return [""] * len(row)

    styled = display_df.style.apply(highlight_best, axis=1).format(
        {"MAE": "{:.2f}", "RMSE": "{:.2f}", "R2 Score": "{:.3f}"}
    )
    st.dataframe(styled, use_container_width=True, height=380)
    st.markdown('<div class="chart-note">Top row (highlighted) is the automatically selected best model.</div>', unsafe_allow_html=True)

    st.markdown("#### Model Comparison Charts")
    plot_df = results_df.copy()
    plot_df["Label"] = plot_df["Model"] + " (" + plot_df["Target"] + ")"

    fig_r2 = px.bar(plot_df.sort_values("R2 Score"), x="R2 Score", y="Label", orientation="h",
                     title="R² Score by Model", color="Target",
                     color_discrete_map={"Raw": "#D1D5DB", "Log": "#E50914"})
    st.plotly_chart(style_plotly(fig_r2), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig_mae = px.bar(plot_df.sort_values("MAE"), x="MAE", y="Label", orientation="h",
                          title="MAE by Model", color="Target",
                          color_discrete_map={"Raw": "#D1D5DB", "Log": "#E50914"})
        st.plotly_chart(style_plotly(fig_mae), use_container_width=True)
    with c2:
        fig_rmse = px.bar(plot_df.sort_values("RMSE"), x="RMSE", y="Label", orientation="h",
                           title="RMSE by Model", color="Target",
                           color_discrete_map={"Raw": "#D1D5DB", "Log": "#E50914"})
        st.plotly_chart(style_plotly(fig_rmse), use_container_width=True)

    st.markdown("#### Why the Best Model Performed Better")
    st.markdown(
        f'<div class="section-card">{family_explanation(artifacts["best_family"], artifacts["best_target"], artifacts["best_model_name"])}</div>',
        unsafe_allow_html=True,
    )

    baseline_r2 = results_df[results_df["Model"] == "Linear Regression"]["R2 Score"].max()
    improvement = artifacts["best_r2"] - baseline_r2
    if improvement > 0.02:
        st.success(
            f"✅ The best model improves R² by {improvement:.3f} over a plain Linear Regression baseline "
            f"({baseline_r2:.3f} → {artifacts['best_r2']:.3f})."
        )
    else:
        st.info(
            f"ℹ️ The best model's R² ({artifacts['best_r2']:.3f}) is close to the plain Linear Regression "
            f"baseline ({baseline_r2:.3f}). This honestly reflects the dataset's structure rather than an "
            "inflated result."
        )

elif page == "🎯 Prediction Simulator":
    st.markdown("## 🎯 Prediction Simulator")
    st.markdown(
        f"This simulator automatically uses the best trained model: "
        f"**{artifacts['best_model_name']} ({artifacts['best_target']} target)**."
    )

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
            inches = st.slider("Screen Size (Inches)", float(df_clean["Inches"].min()),
                                float(df_clean["Inches"].max()), float(df_clean["Inches"].median()))
        with c2:
            resolution = st.selectbox("Screen Resolution", resolutions)
            ram = st.selectbox("RAM (GB)", sorted(df_clean["Ram"].dropna().unique().tolist()))
            weight = st.slider("Weight (kg)", float(df_clean["Weight"].min()),
                                float(df_clean["Weight"].max()), float(df_clean["Weight"].median()))
        with c3:
            memory_gb = st.selectbox("Storage (GB)", sorted(df_clean["Memory_GB"].dropna().unique().tolist()))
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
        input_encoded = input_encoded.reindex(columns=artifacts["X_columns"], fill_value=0)

        try:
            if artifacts["best_scaled"]:
                input_for_model = artifacts["scaler"].transform(input_encoded)
            else:
                input_for_model = input_encoded

            prediction = artifacts["best_wrapper"].predict(input_for_model)[0]

            st.markdown(
                f"""
                <div class="pred-card">
                    <div class="label">Estimated Laptop Price</div>
                    <div class="price">{prediction:,.2f}</div>
                    <div class="modelname">Predicted using {artifacts['best_model_name']} ({artifacts['best_target']} target)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"Based on the selected laptop specifications, the model estimates the price to be "
                f"around **{prediction:,.2f}**."
            )
            st.caption(
                f"This is a data-driven estimate, not a guarantee. On the test set, "
                f"{artifacts['best_model_name']} has a typical error (MAE) of about "
                f"{artifacts['best_mae']:.0f} and an R² of {artifacts['best_r2']:.3f} — use the "
                "prediction as a reference point, not an exact quote."
            )
        except Exception as e:
            st.error(f"⚠️ Could not generate a prediction with the selected inputs. Details: {e}")

elif page == "📈 Actual vs Predicted":
    st.markdown(f"## 📈 Actual vs Predicted Prices — {artifacts['best_model_name']}")

    y_test = artifacts["y_test"]
    y_pred_best = artifacts["best_pred"]

    plot_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred_best})
    min_val = min(plot_df["Actual"].min(), plot_df["Predicted"].min())
    max_val = max(plot_df["Actual"].max(), plot_df["Predicted"].max())

    fig = px.scatter(plot_df, x="Actual", y="Predicted", opacity=0.6,
                      title=f"Actual vs Predicted Laptop Prices — {artifacts['best_model_name']}",
                      color_discrete_sequence=["#E50914"])
    fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode="lines",
                              name="Perfect Prediction", line=dict(color="#D1D5DB", dash="dash")))
    st.plotly_chart(style_plotly(fig), use_container_width=True)
    st.caption("Points closer to the diagonal line indicate predictions closer to the actual price.")

    st.markdown("#### Residual (Error) Distribution")
    residuals = plot_df["Actual"] - plot_df["Predicted"]
    fig2 = px.histogram(residuals, nbins=30, title="Prediction Error Distribution",
                         color_discrete_sequence=["#D1D5DB"])
    fig2.update_layout(showlegend=False, xaxis_title="Actual - Predicted", yaxis_title="Count")
    st.plotly_chart(style_plotly(fig2), use_container_width=True)
    st.caption("A distribution centered near zero with a tight spread indicates consistently small errors.")

elif page == "⭐ Feature Importance":
    st.markdown(f"## ⭐ Feature Importance — {artifacts['best_model_name']}")

    importances = artifacts["importances"]
    if importances is None:
        st.warning("⚠️ Feature importance could not be computed for this model configuration.")
    else:
        top15 = importances.sort_values(ascending=True).tail(15)
        fig = px.bar(top15, x=top15.values, y=top15.index, orientation="h",
                     title="Top 15 Most Important Features (Permutation Importance)",
                     color_discrete_sequence=["#E50914"])
        fig.update_layout(xaxis_title="Importance (drop in R² when shuffled)", yaxis_title="Feature")
        st.plotly_chart(style_plotly(fig), use_container_width=True)
        st.caption(
            "Computed using permutation importance on the test set: each feature is shuffled and the "
            "drop in R² measures how much the model relies on it. This works consistently across all "
            "model types, including ones without a built-in importance score."
        )

    st.markdown(
        """
        <div class="section-card">
        Specifications like RAM, storage capacity, weight, screen size, and certain premium brands or
        CPU/GPU categories tend to have the strongest influence on laptop price, since they directly
        relate to performance and build quality.
        </div>
        """,
        unsafe_allow_html=True,
    )

elif page == "📝 Conclusion":
    st.markdown("## 📝 Conclusion")

    baseline_r2 = results_df[results_df["Model"] == "Linear Regression"]["R2 Score"].max()
    improvement = artifacts["best_r2"] - baseline_r2
    improved_meaningfully = improvement > 0.02

    if improved_meaningfully:
        improvement_line = (
            f"Compared to a plain Linear Regression baseline (R² {baseline_r2:.3f}), the best model "
            f"improved R² by {improvement:.3f}, which is a real, measurable gain."
        )
    else:
        improvement_line = (
            f"Compared to a plain Linear Regression baseline (R² {baseline_r2:.3f}), the best model's "
            f"R² of {artifacts['best_r2']:.3f} is only a small improvement, which honestly suggests this "
            "dataset's pricing pattern is close to linear for the features available."
        )

    st.markdown(
        f"""
        <div class="section-card">
        First, I cleaned the dataset by removing empty rows, duplicated rows, unnecessary columns, and
        outliers. I also converted columns like Ram, Weight, Inches, and Memory into numeric values.
        After that, I prepared the data by encoding text columns and scaling the features for the linear
        models. Then I trained eight different regression models, each once on the raw price and once on
        a log-transformed price, to see which setup would perform best.
        <br><br>
        The best result came from <b>{artifacts['best_model_name']}</b> trained on the
        <b>{artifacts['best_target'].lower()} price target</b>, with an R² score of
        {artifacts['best_r2']:.3f} and an MAE of {artifacts['best_mae']:.2f}. {improvement_line}
        This means {artifacts['best_model_name']} predicted laptop prices more accurately than the
        simpler baseline model.
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card("Final R² Score", f"{artifacts['best_r2']:.3f}", sub=artifacts["best_model_name"])
    with c2:
        kpi_card("Final MAE", f"{artifacts['best_mae']:.2f}", sub=artifacts["best_model_name"])
    with c3:
        kpi_card("Final RMSE", f"{artifacts['best_rmse']:.2f}", sub=artifacts["best_model_name"])
