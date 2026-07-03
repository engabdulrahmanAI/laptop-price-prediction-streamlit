import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Laptop Price Prediction", page_icon="💻", layout="wide")

st.markdown("""
<style>
.stApp {background:#0B0B0F;color:white;}
[data-testid="stHeader"]{background:#0B0B0F;}
section[data-testid="stSidebar"]{background:#111116;}
h1,h2,h3,h4,p,span,label,div{color:white!important;}
.card{background:#16161D;border:1px solid #2A2A35;border-radius:14px;padding:22px;margin-bottom:18px;}
.price{font-size:42px;font-weight:900;color:#fff!important;}
.red{color:#E50914!important;font-weight:800;}
.stButton>button{background:#E50914;color:white;border:0;border-radius:8px;font-weight:800;}
</style>
""", unsafe_allow_html=True)

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "laptopData.csv")

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

def memory_to_gb(x):
    total = 0
    for part in str(x).split("+"):
        size = part.strip().split()[0] if part.strip() else "0"
        try:
            if "TB" in size:
                total += float(size.replace("TB", "")) * 1024
            elif "GB" in size:
                total += float(size.replace("GB", ""))
        except Exception:
            pass
    return total

@st.cache_data
def clean_data(df):
    df = df.copy()
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df = df.dropna(how="all").drop_duplicates()
    df["Ram"] = pd.to_numeric(df["Ram"].astype(str).str.replace("GB", "", regex=False), errors="coerce")
    df["Weight"] = pd.to_numeric(df["Weight"].astype(str).str.replace("kg", "", regex=False), errors="coerce")
    df["Inches"] = pd.to_numeric(df["Inches"], errors="coerce")
    df["Memory_GB"] = df["Memory"].apply(memory_to_gb)
    df["Cpu_Brand"] = df["Cpu"].astype(str).str.split().str[0]
    df["Gpu_Brand"] = df["Gpu"].astype(str).str.split().str[0]
    df = df.drop(columns=["Memory", "Cpu", "Gpu"], errors="ignore")
    df = df.dropna(subset=["Price"])
    for col in ["Ram", "Weight", "Inches", "Memory_GB"]:
        df[col] = df[col].fillna(df[col].median())
    q1, q3 = df["Price"].quantile([0.25, 0.75])
    iqr = q3 - q1
    df = df[(df["Price"] >= q1 - 1.5 * iqr) & (df["Price"] <= q3 + 1.5 * iqr)]
    return df.reset_index(drop=True)

@st.cache_resource
def train_model(df):
    X = pd.get_dummies(df.drop(columns=["Price"]), drop_first=True)
    y = df["Price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=300, max_depth=20, min_samples_leaf=2, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "R2": r2_score(y_test, pred),
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
    }
    return model, X.columns, metrics, y_test, pred

st.sidebar.title("💻 Laptop Price AI")
page = st.sidebar.radio("Pages", ["Home", "EDA", "Predict", "Model", "Conclusion"])

if os.path.exists(DATA_PATH):
    raw = load_data(DATA_PATH)
else:
    file = st.sidebar.file_uploader("Upload laptopData.csv", type="csv")
    if file is None:
        st.warning("Upload laptopData.csv or place it next to app.py")
        st.stop()
    raw = load_data(file)

df = clean_data(raw)
model, columns, metrics, y_test, y_pred = train_model(df)

st.sidebar.success("Model: Random Forest")
st.sidebar.caption(f"R²: {metrics['R2']:.3f}")

if page == "Home":
    st.markdown("""
    <div class='card'>
    <h1>💻 Laptop Price Prediction</h1>
    <p>This Streamlit app predicts laptop prices using machine learning.</p>
    <p>The project includes data cleaning, EDA, model training, prediction, and final conclusion.</p>
    </div>
    """, unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Model", "Random Forest")
    c3.metric("R² Score", f"{metrics['R2']:.3f}")
    st.dataframe(df.head(10), width="stretch")

elif page == "EDA":
    st.title("EDA")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x="Price", title="Price Distribution")
        st.plotly_chart(fig, width="stretch")
    with c2:
        avg = df.groupby("Company")["Price"].mean().sort_values(ascending=False).reset_index()
        fig = px.bar(avg, x="Company", y="Price", title="Average Price by Company")
        st.plotly_chart(fig, width="stretch")
    c3, c4 = st.columns(2)
    with c3:
        avg_ram = df.groupby("Ram")["Price"].mean().reset_index()
        fig = px.bar(avg_ram, x="Ram", y="Price", title="Average Price by RAM")
        st.plotly_chart(fig, width="stretch")
    with c4:
        fig = px.scatter(df, x="Memory_GB", y="Price", title="Storage vs Price")
        st.plotly_chart(fig, width="stretch")

elif page == "Predict":
    st.title("Prediction Simulator")
    st.info("Model currently used: Random Forest")
    with st.form("form"):
        c1, c2, c3 = st.columns(3)
        company = c1.selectbox("Company", sorted(df["Company"].dropna().unique()))
        typename = c1.selectbox("Type", sorted(df["TypeName"].dropna().unique()))
        inches = c1.slider("Screen Size", float(df["Inches"].quantile(0.05)), float(df["Inches"].quantile(0.95)), float(df["Inches"].median()))
        resolution = c2.selectbox("Screen Resolution", sorted(df["ScreenResolution"].dropna().unique()))
        ram_options = [r for r in sorted(df["Ram"].dropna().unique()) if r <= 32]
        ram = c2.selectbox("RAM", ram_options, index=ram_options.index(8) if 8 in ram_options else 0)
        weight = c2.slider("Weight", float(df["Weight"].quantile(0.05)), float(df["Weight"].quantile(0.95)), float(df["Weight"].median()))
        memory = c3.selectbox("Storage", sorted(df["Memory_GB"].dropna().unique()))
        cpu = c3.selectbox("CPU Brand", sorted(df["Cpu_Brand"].dropna().unique()))
        gpu = c3.selectbox("GPU Brand", sorted(df["Gpu_Brand"].dropna().unique()))
        opsys = st.selectbox("Operating System", sorted(df["OpSys"].dropna().unique()))
        submit = st.form_submit_button("Predict Price", width="stretch")
    if submit:
        row = pd.DataFrame([{
            "Company": company, "TypeName": typename, "Inches": inches,
            "ScreenResolution": resolution, "Ram": ram, "OpSys": opsys,
            "Weight": weight, "Memory_GB": memory, "Cpu_Brand": cpu, "Gpu_Brand": gpu,
        }])
        row = pd.get_dummies(row).reindex(columns=columns, fill_value=0)
        price = model.predict(row)[0]
        price = np.clip(price, df["Price"].quantile(0.01), df["Price"].quantile(0.99))
        st.markdown(f"<div class='card'><div>Estimated Dataset Price</div><div class='price'>{price:,.2f}</div></div>", unsafe_allow_html=True)

elif page == "Model":
    st.title("Model Performance")
    c1, c2, c3 = st.columns(3)
    c1.metric("R² Score", f"{metrics['R2']:.3f}")
    c2.metric("MAE", f"{metrics['MAE']:.2f}")
    c3.metric("RMSE", f"{metrics['RMSE']:.2f}")
    result = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
    fig = px.scatter(result, x="Actual", y="Predicted", title="Actual vs Predicted")
    st.plotly_chart(fig, width="stretch")

else:
    st.title("Conclusion")
    st.markdown(f"""
    <div class='card'>
    First, I cleaned the dataset by removing empty rows, duplicated rows, unnecessary columns, and outliers.
    I converted Ram, Weight, Inches, and Memory into numeric values.
    Then I trained a Random Forest Regressor to predict laptop prices.
    The model achieved R² score of <b>{metrics['R2']:.3f}</b>, MAE of <b>{metrics['MAE']:.2f}</b>, and RMSE of <b>{metrics['RMSE']:.2f}</b>.
    Random Forest was used because it gives stable predictions and handles non-linear relationships well.
    </div>
    """, unsafe_allow_html=True)
