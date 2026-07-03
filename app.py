import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Laptop Price Prediction", page_icon="💻", layout="wide", initial_sidebar_state="expanded")

CUSTOM_CSS = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header[data-testid="stHeader"] {background: #0B0B0F;}

.stApp {
    background-color: #0B0B0F;
}
.stApp, .stApp p, .stApp span, .stApp label, .stApp li, .stApp div {
    color: #FFFFFF;
}
::placeholder {
    color: #9CA3AF !important;
    opacity: 1;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

section[data-testid="stSidebar"] {
    background-color: #0B0B0F;
    border-right: 1px solid #2A2A35;
}
section[data-testid="stSidebar"] * {
    color: #FFFFFF;
}

.hero {
    background: linear-gradient(135deg, #16161D 0%, #1F1F2A 55%, #2A0A0E 100%);
    padding: 2.4rem 2.2rem;
    border-radius: 16px;
    margin-bottom: 1.4rem;
    border: 1px solid #2A2A35;
}
.hero .tagline {
    display: inline-block;
    color: #E50914;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-size: 0.78rem;
    margin-bottom: 0.6rem;
}
.hero h1 {
    color: #FFFFFF !important;
    font-size: 2.2rem;
    font-weight: 800;
    margin-bottom: 0.5rem;
}
.hero p {
    color: #D1D5DB !important;
    font-size: 1.02rem;
    max-width: 780px;
    line-height: 1.6;
}

.card {
    background: #16161D;
    border: 1px solid #2A2A35;
    border-radius: 12px;
    padding: 1.3rem 1.4rem;
    margin-bottom: 1rem;
}
.card p, .card li, .card b {
    color: #D1D5DB !important;
}
.card b {
    color: #FFFFFF !important;
}

.kpi {
    background: #16161D;
    border: 1px solid #2A2A35;
    border-radius: 12px;
    padding: 1.1rem 1.2rem;
}
.kpi .label {
    color: #9CA3AF !important;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.3rem;
}
.kpi .value {
    color: #FFFFFF !important;
    font-size: 1.6rem;
    font-weight: 800;
}
.kpi .sub {
    color: #E50914 !important;
    font-size: 0.78rem;
    margin-top: 0.2rem;
    font-weight: 600;
}

.pred-card {
    background: linear-gradient(135deg, #1F1F2A 0%, #2A0A0E 100%);
    border: 1px solid #E50914;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.pred-card .label {
    color: #D1D5DB !important;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.5rem;
}
.pred-card .price {
    color: #FFFFFF !important;
    font-size: 2.6rem;
    font-weight: 800;
}
.pred-card .modelname {
    color: #E50914 !important;
    font-weight: 700;
    font-size: 0.88rem;
    margin-top: 0.5rem;
}

div[data-testid="stMetricValue"] { color: #FFFFFF; }
div[data-testid="stMetricLabel"] { color: #9CA3AF; }

div[data-baseweb="select"] > div {
    background-color: #16161D !important;
    border-color: #2A2A35 !important;
    color: #FFFFFF !important;
}
ul[data-baseweb="menu"] {
    background-color: #16161D !important;
}
li[role="option"] {
    color: #FFFFFF !important;
}
li[role="option"]:hover {
    background-color: #2A2A35 !important;
}

div[data-testid="stSlider"] div[data-baseweb="slider"] > div {
    background: #2A2A35 !important;
}
div[data-testid="stSlider"] div[role="slider"] {
    background-color: #E50914 !important;
    border-color: #E50914 !important;
}
div[data-testid="stTickBar"] {
    color: #9CA3AF !important;
}

.stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {
    background-color: #E50914;
    color: #FFFFFF !important;
    border: none;
    border-radius: 8px;
    font-weight: 700;
}
.stButton > button:hover, .stFormSubmitButton > button:hover, .stDownloadButton > button:hover {
    background-color: #B20710;
    color: #FFFFFF !important;
}

button[data-baseweb="tab"] {
    color: #9CA3AF !important;
    background-color: transparent !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #FFFFFF !important;
}
div[data-baseweb="tab-highlight"] {
    background-color: #E50914 !important;
}
div[data-baseweb="tab-border"] {
    background-color: #2A2A35 !important;
}

section[data-testid="stFileUploaderDropzone"] {
    background-color: #16161D !important;
    border: 1px dashed #2A2A35 !important;
}
section[data-testid="stFileUploaderDropzone"] * {
    color: #FFFFFF !important;
}
section[data-testid="stFileUploaderDropzone"] button {
    background-color: #E50914 !important;
    color: #FFFFFF !important;
    border: none !important;
}

table {
    background-color: #16161D !important;
    color: #FFFFFF !important;
}
th {
    background-color: #1F1F2A !important;
    color: #FFFFFF !important;
}
td {
    color: #FFFFFF !important;
    border-color: #2A2A35 !important;
}

hr { border-color: #2A2A35; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

PLOT_BG = "#0B0B0F"
CARD_BG = "#16161D"
ACCENT = "#E50914"
MUTED = "#D1D5DB"
GRID = "#2A2A35"


def style_fig(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        font_color="#FFFFFF",
        title_font_color="#FFFFFF",
        legend_font_color="#FFFFFF",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, color=MUTED)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, color=MUTED)
    return fig


def style_table(df):
    return df.style.set_properties(**{
        "background-color": CARD_BG, "color": "#FFFFFF", "border-color": GRID,
    }).set_table_styles([
        {"selector": "th", "props": [("background-color", "#1F1F2A"), ("color", "#FFFFFF")]}
    ])


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
            total += float(size.replace("TB", "")) * 1024
        elif "GB" in size:
            total += float(size.replace("GB", ""))
    return total


@st.cache_data(show_spinner=False)
def clean_data(df_raw):
    df = df_raw.copy()
    shape_before = df.shape

    if "Unnamed: 0" in df.columns:
        df = df.drop("Unnamed: 0", axis=1)

    df = df.dropna(how="all")
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
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df = df[(df["Price"] >= lower) & (df["Price"] <= upper)]

    df = df.reset_index(drop=True)
    stats = {"shape_before": shape_before, "shape_after": df.shape}
    return df, stats


def build_registry():
    return {
        "Linear Regression": {"model": LinearRegression(), "scaled": True},
        "Ridge Regression": {"model": Ridge(alpha=1.0, random_state=42), "scaled": True},
        "Random Forest": {
            "model": RandomForestRegressor(n_estimators=300, max_depth=20, min_samples_split=4,
                                            min_samples_leaf=2, random_state=42, n_jobs=-1),
            "scaled": False,
        },
        "Extra Trees": {
            "model": ExtraTreesRegressor(n_estimators=300, max_depth=20, min_samples_split=4,
                                          min_samples_leaf=2, random_state=42, n_jobs=-1),
            "scaled": False,
        },
        "Gradient Boosting": {
            "model": GradientBoostingRegressor(n_estimators=300, max_depth=4, min_samples_split=4,
                                                min_samples_leaf=2, learning_rate=0.05, random_state=42),
            "scaled": False,
        },
    }


class FittedWrapper:
    def __init__(self, model, is_log):
        self.model = model
        self.is_log = is_log

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

    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    y_train_log = np.log1p(y_train)

    registry = build_registry()
    rows = []
    trained = {}

    for name, cfg in registry.items():
        for target in ["Raw", "Log"]:
            model = clone(cfg["model"])
            X_tr = X_train_scaled if cfg["scaled"] else X_train
            X_te = X_test_scaled if cfg["scaled"] else X_test
            y_tr = y_train_log if target == "Log" else y_train

            model.fit(X_tr, y_tr)
            wrapper = FittedWrapper(model, is_log=(target == "Log"))
            pred = wrapper.predict(X_te)

            mae = mean_absolute_error(y_test, pred)
            rmse = np.sqrt(mean_squared_error(y_test, pred))
            r2 = r2_score(y_test, pred)

            key = f"{name} ({target})"
            rows.append({"Key": key, "Model": name, "Target": target, "MAE": mae, "RMSE": rmse, "R2 Score": r2})
            trained[key] = {"wrapper": wrapper, "scaled": cfg["scaled"], "pred": pred}

    results_df = pd.DataFrame(rows).sort_values(by=["R2 Score", "RMSE"], ascending=[False, True]).reset_index(drop=True)

    stable_candidates = results_df[results_df["Model"].isin(["Random Forest", "Extra Trees"])]
    stable_candidates = stable_candidates.sort_values(by=["R2 Score", "RMSE"], ascending=[False, True])
    stable_key = stable_candidates.iloc[0]["Key"]
    stable_row = stable_candidates.iloc[0]
    stable = trained[stable_key]

    price_low = float(y.quantile(0.01))
    price_high = float(y.quantile(0.99))

    return {
        "results_df": results_df,
        "model_name": stable_row["Model"],
        "target_mode": stable_row["Target"],
        "r2": stable_row["R2 Score"],
        "mae": stable_row["MAE"],
        "rmse": stable_row["RMSE"],
        "wrapper": stable["wrapper"],
        "scaled": stable["scaled"],
        "scaler": scaler,
        "X_columns": X_encoded.columns,
        "y_test": y_test,
        "pred": stable["pred"],
        "price_low": price_low,
        "price_high": price_high,
    }


def kpi(label, value, sub=None):
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    st.markdown(f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div>{sub_html}</div>', unsafe_allow_html=True)


st.sidebar.markdown("## 💻 Laptop Price AI")
st.sidebar.markdown("---")

df_raw = None
data_source = None

if os.path.exists(DATA_PATH):
    df_raw = load_raw_data(DATA_PATH)
    data_source = "built-in"
    st.sidebar.success("Using built-in laptopData.csv")
else:
    st.sidebar.warning("laptopData.csv not found in app folder.")
    uploaded = st.sidebar.file_uploader("Upload laptopData.csv", type=["csv"], width="stretch")
    if uploaded is not None:
        df_raw = load_raw_data(uploaded)
        data_source = "uploaded"

PAGES = ["🏠 Home", "🔍 EDA", "🤖 Model Performance", "🎯 Prediction Simulator", "📝 Conclusion"]
page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")
st.sidebar.markdown("---")

if df_raw is None:
    st.markdown(
        """
        <div class="hero">
            <div class="tagline">Machine Learning Portfolio Project</div>
            <h1>💻 Laptop Price Prediction</h1>
            <p>Add <b>laptopData.csv</b> next to app.py, or upload it from the sidebar, to explore the
            dataset, compare models, and predict laptop prices.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.warning("No dataset available. Add laptopData.csv to the app folder or upload it from the sidebar.")
    st.stop()

df_clean, clean_stats = clean_data(df_raw)
artifacts = train_models(df_clean)
results_df = artifacts["results_df"]

st.sidebar.caption(f"Prediction engine: **{artifacts['model_name']} ({artifacts['target_mode']})**")
st.sidebar.caption(f"R² {artifacts['r2']:.3f} · MAE {artifacts['mae']:.0f} · RMSE {artifacts['rmse']:.0f}")

if page == "🏠 Home":
    st.markdown(
        """
        <div class="hero">
            <div class="tagline">Machine Learning Portfolio Project</div>
            <h1>💻 Laptop Price Prediction</h1>
            <p>Predicting laptop prices from technical specifications using supervised machine learning,
            trained automatically on laptopData.csv.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("Dataset Rows", f"{clean_stats['shape_after'][0]:,}")
    with c2:
        kpi("Prediction Model", artifacts["model_name"], sub=f"{artifacts['target_mode']} target")
    with c3:
        kpi("R² Score", f"{artifacts['r2']:.3f}")
    with c4:
        kpi("MAE", f"{artifacts['mae']:.0f}")

    st.markdown(
        """
        <div class="card">
        Laptop prices depend on brand, processor, RAM, storage, screen size, and GPU, which makes manual
        pricing inconsistent. This model learns those relationships from real listings to produce a
        consistent, data-driven price estimate.
        </div>
        """,
        unsafe_allow_html=True,
    )

elif page == "🔍 EDA":
    st.markdown("## 🔍 Exploratory Data Analysis")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df_clean, x="Price", nbins=30, title="Price Distribution", color_discrete_sequence=[ACCENT])
        st.plotly_chart(style_fig(fig), width="stretch")
    with c2:
        avg_price = df_clean.groupby("Company")["Price"].mean().sort_values(ascending=False).reset_index()
        fig = px.bar(avg_price, x="Company", y="Price", title="Average Price by Company", color_discrete_sequence=[ACCENT])
        st.plotly_chart(style_fig(fig), width="stretch")

    c3, c4 = st.columns(2)
    with c3:
        avg_ram = df_clean.groupby("Ram")["Price"].mean().sort_index().reset_index()
        fig = px.bar(avg_ram, x="Ram", y="Price", title="Average Price by RAM (GB)", color_discrete_sequence=[MUTED])
        st.plotly_chart(style_fig(fig), width="stretch")
    with c4:
        numeric_df = df_clean.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        fig = px.imshow(corr, text_auto=".2f", aspect="auto", title="Correlation Heatmap", color_continuous_scale="RdBu_r")
        st.plotly_chart(style_fig(fig), width="stretch")

elif page == "🤖 Model Performance":
    st.markdown("## 🤖 Model Performance")

    display_df = results_df[["Model", "Target", "MAE", "RMSE", "R2 Score"]].copy()

    def highlight(row):
        if row["Model"] == artifacts["model_name"] and row["Target"] == artifacts["target_mode"]:
            return ["background-color: rgba(229,9,20,0.22)"] * len(row)
        return [""] * len(row)

    styled = display_df.style.apply(highlight, axis=1).format({"MAE": "{:.0f}", "RMSE": "{:.0f}", "R2 Score": "{:.3f}"})
    st.dataframe(styled, width="stretch", height=280)

    st.markdown(
        f"""
        <div class="card">
        <b>{artifacts['model_name']} ({artifacts['target_mode']} target)</b> is used for predictions.
        Random Forest and Extra Trees are prioritized over models with a marginally higher R² because
        tree ensembles produce more stable, realistic predictions on new inputs instead of extrapolating
        into extreme values.
        </div>
        """,
        unsafe_allow_html=True,
    )

    plot_df = results_df.copy()
    plot_df["Label"] = plot_df["Model"] + " (" + plot_df["Target"] + ")"
    fig = px.bar(plot_df.sort_values("R2 Score"), x="R2 Score", y="Label", orientation="h",
                 title="R² Score by Model", color="Target", color_discrete_map={"Raw": MUTED, "Log": ACCENT})
    st.plotly_chart(style_fig(fig), width="stretch")

    y_test = artifacts["y_test"]
    pred = artifacts["pred"]
    plot_df2 = pd.DataFrame({"Actual": y_test.values, "Predicted": pred})
    lo = min(plot_df2["Actual"].min(), plot_df2["Predicted"].min())
    hi = max(plot_df2["Actual"].max(), plot_df2["Predicted"].max())
    fig2 = px.scatter(plot_df2, x="Actual", y="Predicted", opacity=0.6, title="Actual vs Predicted",
                       color_discrete_sequence=[ACCENT])
    fig2.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect Prediction",
                               line=dict(color=MUTED, dash="dash")))
    st.plotly_chart(style_fig(fig2), width="stretch")

elif page == "🎯 Prediction Simulator":
    st.markdown("## 🎯 Prediction Simulator")
    st.markdown(f"Using **{artifacts['model_name']} ({artifacts['target_mode']} target)** for a stable, realistic estimate.")

    companies = sorted(df_clean["Company"].dropna().unique().tolist())
    types = sorted(df_clean["TypeName"].dropna().unique().tolist())
    resolutions = sorted(df_clean["ScreenResolution"].dropna().unique().tolist())
    opsys = sorted(df_clean["OpSys"].dropna().unique().tolist())
    cpu_brands = sorted(df_clean["Cpu_Brand"].dropna().unique().tolist())
    gpu_brands = sorted(df_clean["Gpu_Brand"].dropna().unique().tolist())

    ram_values = sorted(df_clean["Ram"].dropna().unique().tolist())
    high_ram_count = (df_clean["Ram"] > 32).sum()
    if high_ram_count / len(df_clean) < 0.03:
        ram_values = [r for r in ram_values if r <= 32]

    inches_low, inches_high = df_clean["Inches"].quantile([0.05, 0.95])
    weight_low, weight_high = df_clean["Weight"].quantile([0.05, 0.95])
    memory_values = sorted(df_clean["Memory_GB"].dropna().unique().tolist())

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            company = st.selectbox("Company", companies)
            typename = st.selectbox("Type", types)
            inches = st.slider("Screen Size (Inches)", float(inches_low), float(inches_high), float(df_clean["Inches"].median()))
        with c2:
            resolution = st.selectbox("Screen Resolution", resolutions)
            ram = st.selectbox("RAM (GB)", ram_values)
            weight = st.slider("Weight (kg)", float(weight_low), float(weight_high), float(df_clean["Weight"].median()))
        with c3:
            memory_gb = st.selectbox("Storage (GB)", memory_values)
            cpu_brand = st.selectbox("CPU Brand", cpu_brands)
            gpu_brand = st.selectbox("GPU Brand", gpu_brands)
        opsys_choice = st.selectbox("Operating System", opsys)

        submitted = st.form_submit_button("🔮 Predict Price", width="stretch")

    if submitted:
        input_dict = {
            "Company": company, "TypeName": typename, "Inches": inches, "ScreenResolution": resolution,
            "Ram": ram, "OpSys": opsys_choice, "Weight": weight, "Memory_GB": memory_gb,
            "Cpu_Brand": cpu_brand, "Gpu_Brand": gpu_brand,
        }
        input_df = pd.DataFrame([input_dict])
        input_encoded = pd.get_dummies(input_df)
        input_encoded = input_encoded.reindex(columns=artifacts["X_columns"], fill_value=0)

        try:
            if artifacts["scaled"]:
                input_for_model = artifacts["scaler"].transform(input_encoded)
            else:
                input_for_model = input_encoded

            prediction = artifacts["wrapper"].predict(input_for_model)[0]
            prediction = float(np.clip(prediction, artifacts["price_low"], artifacts["price_high"]))

            st.markdown(
                f"""
                <div class="pred-card">
                    <div class="label">Estimated Dataset Price</div>
                    <div class="price">{prediction:,.2f}</div>
                    <div class="modelname">Predicted using {artifacts['model_name']} ({artifacts['target_mode']} target)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption("The price unit depends on the currency used in the original dataset. This is an estimate based on historical listings, not a live market price.")
        except Exception as e:
            st.error(f"Could not generate a prediction with the selected inputs. Details: {e}")

elif page == "📝 Conclusion":
    st.markdown("## 📝 Conclusion")
    st.markdown(
        f"""
        <div class="card">
        I cleaned the dataset by removing empty rows, duplicates, and price outliers, then converted Ram,
        Weight, Inches, and Memory into numeric values and encoded the categorical columns. I trained five
        models, each on both the raw and log-transformed price, and compared them using MAE, RMSE, and R².
        <br><br>
        For predictions, I chose <b>{artifacts['model_name']} ({artifacts['target_mode']} target)</b>
        instead of automatically taking the highest R² score, since tree ensembles like Random Forest and
        Extra Trees give more stable, realistic prices instead of occasionally extreme ones. This model
        reached an R² of {artifacts['r2']:.3f} with an MAE of {artifacts['mae']:.0f} on the test set.
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        kpi("Final R²", f"{artifacts['r2']:.3f}")
    with c2:
        kpi("Final MAE", f"{artifacts['mae']:.0f}")
    with c3:
        kpi("Final RMSE", f"{artifacts['rmse']:.0f}")
