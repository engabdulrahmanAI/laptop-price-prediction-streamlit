# 💻 Laptop Price Prediction using Machine Learning

An interactive Streamlit dashboard that predicts laptop prices from technical specifications, built on
top of a data cleaning and modeling pipeline developed in a Jupyter Notebook.

## Project Summary

- **Dataset:** `laptopData.csv`
- **Target column:** `Price`
- **Rows after cleaning:** 1216
- **Models:** Linear Regression, Random Forest Regressor
- **Best model:** Random Forest Regressor

| Model | MAE | RMSE | R2 Score |
|---|---|---|---|
| Linear Regression | 13663.47 | 18239.40 | 0.691 |
| Random Forest Regressor | 10964.34 | 15227.34 | 0.785 |

## Cleaning Pipeline

- Removed fully empty rows and duplicated rows
- Dropped the unnecessary `Unnamed: 0` column
- Converted `Ram`, `Weight`, and `Inches` from text to numeric
- Parsed `Memory` into a new numeric column `Memory_GB`
- Extracted `Cpu_Brand` and `Gpu_Brand` from the CPU/GPU text columns
- Removed price outliers using the IQR method
- One-hot encoded categorical columns
- Scaled features and applied PCA (95% variance retained) before Linear Regression

## App Pages

1. **Home** — project overview and key numbers
2. **Dataset Overview** — preview, shape, columns, dtypes, missing values
3. **Data Cleaning** — before/after summary and explanations
4. **EDA** — interactive Plotly visualizations
5. **Model Training** — workflow explanation and model comparison
6. **Prediction Simulator** — enter specs and get a price estimate
7. **Actual vs Predicted** — scatter plot and residual distribution
8. **Feature Importance** — top 15 features from Random Forest
9. **Conclusion** — summary of findings

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL shown in the terminal (typically `http://localhost:8501`), and upload
`laptopData.csv` from the sidebar when prompted.

## Notes

- The app trains both models at runtime from the uploaded CSV, using `st.cache_data` and
  `st.cache_resource` so cleaning and training only run once per uploaded file.
- The Prediction Simulator uses the Random Forest model, aligning user input columns to the exact
  one-hot encoded feature set used during training.
- No columns or metrics are invented — all figures come directly from the uploaded dataset and the
  pipeline logic ported from the original notebook.
