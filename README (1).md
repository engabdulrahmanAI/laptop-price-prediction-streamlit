# 💻 Laptop Price Prediction using Machine Learning

An interactive Streamlit dashboard that predicts laptop prices from technical specifications. It
automatically loads `laptopData.csv` from the project folder, compares 8 regression models (each
trained on both the raw and log-transformed target), and always shows real metrics — nothing here is
hardcoded or faked.

## Dataset Loading

- If `laptopData.csv` sits next to `app.py`, it loads automatically — no upload needed.
- If the file is missing, the sidebar shows a file uploader as a fallback so the app never crashes.
- The sidebar always tells you whether you're using the built-in file or an uploaded one.

## Models Compared

Every model below is trained once on the raw `Price` and once on `log1p(Price)` (with predictions
converted back using `expm1`), for 16 total training runs:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- Extra Trees Regressor
- HistGradientBoostingRegressor

Linear-family models (Linear/Ridge/Lasso) train on `StandardScaler`-scaled features; tree-based models
train on the raw one-hot encoded features, since scaling doesn't affect them.

**The best model is chosen automatically** — highest R² Score, ties broken by lowest RMSE — using the
real test-set metrics computed after training. The Model Training page shows the full ranking table so
you can see exactly how every model compares, not just the winner.

## Cleaning Pipeline

- Removed fully empty rows and duplicated rows
- Dropped the unnecessary `Unnamed: 0` column
- Converted `Ram`, `Weight`, and `Inches` from text to numeric
- Parsed `Memory` into a new numeric column `Memory_GB`
- Extracted `Cpu_Brand` and `Gpu_Brand` from the CPU/GPU text columns
- Removed price outliers using the IQR method
- One-hot encoded categorical columns, with the exact same columns reused at prediction time

## App Pages

1. **Home** — project overview and key numbers, all pulled live from the best trained model
2. **Dataset Overview** — preview, shape, columns, dtypes, missing values
3. **Data Cleaning** — before/after summary and explanations
4. **EDA** — interactive Plotly visualizations with short explanations under each chart
5. **Model Training** — full ranking table across all 16 model/target combinations, comparison charts,
   and an explanation of why the winning model performed better
6. **Prediction Simulator** — enter specs and get a price estimate from the best model, with the model
   name shown and an honest note about typical error margin
7. **Actual vs Predicted** — scatter plot and residual distribution for the best model
8. **Feature Importance** — top 15 features via permutation importance, which works for any model type
9. **Conclusion** — summary of findings, written to reflect whichever model actually won

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Place `laptopData.csv` in the same folder as `app.py` before running, then open the local URL shown in
the terminal (typically `http://localhost:8501`).

## Notes

- All metrics on every page come directly from the trained models on the current dataset — none are
  hardcoded. If you swap in a different `laptopData.csv`, the numbers update accordingly.
- If the improved pipeline doesn't beat the old baseline by much, the Model Training and Conclusion
  pages say so honestly instead of overstating the result.
- `st.cache_data` and `st.cache_resource` keep cleaning and training from re-running on every click.
