import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="Laptop Price Prediction",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    html, body, [class*="css"] {
        font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
    }

    .main {
        background-color: #0f1116;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        background: linear-gradient(135deg, #1f2b45 0%, #2c3e6b 50%, #4a2c6b 100%);
        padding: 2.6rem 2.4rem;
        border-radius: 18px;
        margin-bottom: 1.6rem;
        border: 1px solid rgba(255,255,255,0.06);
    }
    .hero h1 {
        color: #ffffff;
        font-size: 2.3rem;
        font-weight: 800;
        margin-bottom: 0.4rem;
    }
    .hero p {
        color: #cfd6e6;
        font-size: 1.05rem;
        max-width: 780px;
        line-height: 1.55;
    }

    .kpi-card {
        background: #161a24;
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 14px;
        padding: 1.2rem 1.3rem;
        text-align: left;
        box-shadow: 0 4px 14px rgba(0,0,0,0.25);
    }
    .kpi-label {
        color: #9aa4b8;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.35rem;
    }
    .kpi-value {
        color: #ffffff;
        font-size: 1.7rem;
        font-weight: 700;
    }
    .kpi-sub {
        color: #6bd0a3;
        font-size: 0.8rem;
        margin-top: 0.2rem;
    }

    .section-card {
        background: #161a24;
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 14px;
        padding: 1.5rem 1.6rem;
        margin-bottom: 1rem;
    }

    .pred-card {
        background: linear-gradient(135deg, #1e3a2e 0%, #16593f 100%);
        border-radius: 18px;
        padding: 2rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .pred-card .price {
        font-size: 2.6rem;
        font-weight: 800;
        color: #ffffff;
    }
    .pred-card .label {
        color: #bfe8d4;
        font-size: 0.95rem;
        margin-bottom: 0.6rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.6rem;
    }

    .badge {
        display: inline-block;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 0.4rem;
    }
    .badge-best {
        background: rgba(107, 208, 163, 0.15);
        color: #6bd0a3;
        border: 1px solid rgba(107,208,163,0.35);
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

NUMERIC_COLS = ["Inches", "Ram", "Weight", "Memory_GB"]
CATEGORICAL_COLS = ["Company", "TypeName", "ScreenResolution", "OpSys", "Cpu_Brand", "Gpu_Brand"]


@st.cache_data(show_spinner=False)
def load_raw_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    return df


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


@st.cache_resource(show_spinner=False)
def train_models(df):
    X = df.drop("Price", axis=1)
    y = df["Price"]
    X_encoded = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )

    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    lr_model = LinearRegression()
    lr_model.fit(X_train_pca, y_train)
    y_pred_lr = lr_model.predict(X_test_pca)

    def metrics(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2 Score": r2}

    results = {
        "Linear Regression": metrics(y_test, y_pred_lr),
        "Random Forest": metrics(y_test, y_pred_rf),
    }

    return {
        "rf_model": rf_model,
        "lr_model": lr_model,
        "scaler": scaler,
        "pca": pca,
        "X_columns": X_encoded.columns,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred_rf": y_pred_rf,
        "y_pred_lr": y_pred_lr,
        "results": results,
        "pca_components_before": X_train.shape[1],
        "pca_components_after": X_train_pca.shape[1],
        "pca_variance": float(pca.explained_variance_ratio_.sum()),
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


st.sidebar.markdown("## 💻 Laptop Price AI")
st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader("Upload laptopData.csv", type=["csv"])

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
st.sidebar.caption("Random Forest Regressor is the selected best model based on R² score and error metrics.")

if uploaded_file is None:
    st.markdown(
        """
        <div class="hero">
            <h1>💻 Laptop Price Prediction using Machine Learning</h1>
            <p>Upload <b>laptopData.csv</b> from the sidebar to explore the dataset, review the cleaning
            pipeline, run the exploratory analysis, train the models, and predict laptop prices.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.warning("⚠️ No dataset uploaded yet. Please upload **laptopData.csv** from the sidebar to continue.")
    st.stop()

df_raw = load_raw_data(uploaded_file)
df_clean, clean_stats = clean_data(df_raw)
artifacts = train_models(df_clean)

results_df = pd.DataFrame(artifacts["results"]).T.reset_index().rename(columns={"index": "Model"})

if page == "🏠 Home":
    st.markdown(
        """
        <div class="hero">
            <h1>💻 Laptop Price Prediction using Machine Learning</h1>
            <p>This project predicts laptop prices from their technical specifications using supervised
            machine learning. It supports buyers, sellers, and retailers who need a fast, data-driven
            estimate of a fair laptop price instead of relying on guesswork.</p>
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
        kpi_card("Best Model", "Random Forest")
    with c4:
        kpi_card("Best R² Score", f"{artifacts['results']['Random Forest']['R2 Score']:.3f}")
    with c5:
        kpi_card("Best MAE", f"{artifacts['results']['Random Forest']['MAE']:.2f}")

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
        11. Scaled features and applied PCA for the Linear Regression model
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
            fig = px.bar(
                df_clean["Company"].value_counts().reset_index(),
                x="Company", y="count", title="Laptop Count by Company",
                color_discrete_sequence=["#6bd0a3"],
            )
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.bar(
                df_clean["TypeName"].value_counts().reset_index(),
                x="TypeName", y="count", title="Laptop Count by Type",
                color_discrete_sequence=["#7c9fe0"],
            )
            st.plotly_chart(fig, use_container_width=True)

        fig = px.histogram(df_clean, x="Price", nbins=30, title="Price Distribution",
                            color_discrete_sequence=["#c084e8"])
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            avg_price_company = df_clean.groupby("Company")["Price"].mean().sort_values(ascending=False).reset_index()
            fig = px.bar(avg_price_company, x="Company", y="Price", title="Average Price by Company",
                         color_discrete_sequence=["#6bd0a3"])
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            avg_price_ram = df_clean.groupby("Ram")["Price"].mean().sort_index().reset_index()
            fig = px.bar(avg_price_ram, x="Ram", y="Price", title="Average Price by RAM (GB)",
                         color_discrete_sequence=["#7c9fe0"])
            st.plotly_chart(fig, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            avg_price_cpu = df_clean.groupby("Cpu_Brand")["Price"].mean().sort_values(ascending=False).reset_index()
            fig = px.bar(avg_price_cpu, x="Cpu_Brand", y="Price", title="Average Price by CPU Brand",
                         color_discrete_sequence=["#c084e8"])
            st.plotly_chart(fig, use_container_width=True)
        with c4:
            avg_price_gpu = df_clean.groupby("Gpu_Brand")["Price"].mean().sort_values(ascending=False).reset_index()
            fig = px.bar(avg_price_gpu, x="Gpu_Brand", y="Price", title="Average Price by GPU Brand",
                         color_discrete_sequence=["#e0a37c"])
            st.plotly_chart(fig, use_container_width=True)

        c5, c6 = st.columns(2)
        with c5:
            fig = px.scatter(df_clean, x="Memory_GB", y="Price", title="Memory (GB) vs Price",
                              opacity=0.6, color_discrete_sequence=["#6bd0a3"])
            st.plotly_chart(fig, use_container_width=True)
        with c6:
            fig = px.scatter(df_clean, x="Weight", y="Price", title="Weight vs Price",
                              opacity=0.6, color_discrete_sequence=["#7c9fe0"])
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        numeric_df = df_clean.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        fig = px.imshow(corr, text_auto=".2f", aspect="auto", title="Correlation Heatmap (Numeric Columns)",
                         color_continuous_scale="RdBu_r")
        st.plotly_chart(fig, use_container_width=True)

elif page == "🤖 Model Training":
    st.markdown("## 🤖 Model Training")

    st.markdown("#### Machine Learning Workflow")
    st.markdown(
        """
        <div class="section-card">
        1. Split the cleaned data into features (X) and target (y = Price)<br>
        2. One-hot encode categorical columns<br>
        3. Split into training and test sets (80/20)<br>
        4. Train <b>Random Forest Regressor</b> directly on the encoded features<br>
        5. For <b>Linear Regression</b>, scale features with StandardScaler, then reduce dimensionality
        with PCA (95% variance retained) before training<br>
        6. Evaluate both models on the same test set using MAE, MSE, RMSE, and R² Score
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        kpi_card("Features Before PCA", artifacts["pca_components_before"])
    with c2:
        kpi_card("Features After PCA", artifacts["pca_components_after"],
                  sub=f"{artifacts['pca_variance']*100:.1f}% variance retained")

    st.markdown("#### Model Performance Comparison")
    styled = results_df.style.format({"MAE": "{:.2f}", "MSE": "{:.2f}", "RMSE": "{:.2f}", "R2 Score": "{:.3f}"})
    st.dataframe(styled, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(results_df, x="Model", y="R2 Score", title="R² Score by Model",
                     color="Model", color_discrete_sequence=["#7c9fe0", "#6bd0a3"])
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        err_df = results_df.melt(id_vars="Model", value_vars=["MAE", "RMSE"], var_name="Metric", value_name="Value")
        fig = px.bar(err_df, x="Metric", y="Value", color="Model", barmode="group",
                     title="Error Metrics by Model", color_discrete_sequence=["#7c9fe0", "#6bd0a3"])
        st.plotly_chart(fig, use_container_width=True)

    st.success("✅ Random Forest Regressor achieves the higher R² score and lower error values, "
               "making it the best model for this dataset.")

elif page == "🎯 Prediction Simulator":
    st.markdown("## 🎯 Prediction Simulator")
    st.markdown("Enter laptop specifications below to get an estimated price from the trained Random Forest model.")

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
            prediction = artifacts["rf_model"].predict(input_encoded)[0]
            st.markdown(
                f"""
                <div class="pred-card">
                    <div class="label">Estimated Laptop Price</div>
                    <div class="price">{prediction:,.2f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"Based on the selected laptop specifications, the model estimates the price to be "
                f"around **{prediction:,.2f}**."
            )
        except Exception as e:
            st.error(f"⚠️ Could not generate a prediction with the selected inputs. Details: {e}")

elif page == "📈 Actual vs Predicted":
    st.markdown("## 📈 Actual vs Predicted Prices — Random Forest")

    y_test = artifacts["y_test"]
    y_pred_rf = artifacts["y_pred_rf"]

    plot_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred_rf})
    min_val = min(plot_df["Actual"].min(), plot_df["Predicted"].min())
    max_val = max(plot_df["Actual"].max(), plot_df["Predicted"].max())

    fig = px.scatter(plot_df, x="Actual", y="Predicted", opacity=0.6,
                      title="Actual vs Predicted Laptop Prices", color_discrete_sequence=["#6bd0a3"])
    fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode="lines",
                              name="Perfect Prediction", line=dict(color="#e07c7c", dash="dash")))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Points closer to the diagonal line indicate predictions closer to the actual price.")

    st.markdown("#### Residual (Error) Distribution")
    residuals = plot_df["Actual"] - plot_df["Predicted"]
    fig2 = px.histogram(residuals, nbins=30, title="Prediction Error Distribution",
                         color_discrete_sequence=["#7c9fe0"])
    fig2.update_layout(showlegend=False, xaxis_title="Actual - Predicted", yaxis_title="Count")
    st.plotly_chart(fig2, use_container_width=True)

elif page == "⭐ Feature Importance":
    st.markdown("## ⭐ Feature Importance — Random Forest")

    importances = pd.Series(artifacts["rf_model"].feature_importances_, index=artifacts["X_columns"])
    top15 = importances.sort_values(ascending=True).tail(15)

    fig = px.bar(top15, x=top15.values, y=top15.index, orientation="h",
                 title="Top 15 Most Important Features", color_discrete_sequence=["#6bd0a3"])
    fig.update_layout(xaxis_title="Importance", yaxis_title="Feature")
    st.plotly_chart(fig, use_container_width=True)

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
    st.markdown(
        """
        <div class="section-card">
        First, I cleaned the dataset by removing empty rows, duplicated rows, unnecessary columns, and
        outliers. I also converted columns like Ram, Weight, Inches, and Memory into numeric values.
        After that, I prepared the data by encoding text columns and scaling the features when needed.
        Then I trained two models: Linear Regression and Random Forest Regressor. Random Forest
        performed better because it had a higher R2 score and lower error values. This means Random
        Forest predicted laptop prices more accurately.
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        kpi_card("Final R² Score", f"{artifacts['results']['Random Forest']['R2 Score']:.3f}", sub="Random Forest")
    with c2:
        kpi_card("Final MAE", f"{artifacts['results']['Random Forest']['MAE']:.2f}", sub="Random Forest")
