# 📡 Telecom Customer Churn Prediction & EDA Dashboard

A clean, beginner-friendly, end-to-end Data Science project built with **Pandas**, **NumPy**, **Matplotlib**, **Seaborn**, **Scikit-Learn**, **Jupyter**, and **Streamlit**.

---

## 🎯 Project Overview

Customer churn (when subscribers cancel service) directly impacts recurring telecom revenue. This project walks through the complete analytics and machine learning workflow:

1. **Synthetic Telecom Data Generation & Cleaning (`data/generate_data.py`)**  
   Generates 5,000 realistic telecom customer records with ~3% injected missing values (`monthly_charges`, `internet_service`, `support_calls`) and demonstrates median/mode imputation.
2. **Exploratory Data Analysis & Model Comparison (`notebooks/EDA_and_Modeling.ipynb`)**  
   Explores churn drivers via visualizations and correlation heatmaps, then compares **Logistic Regression** vs. **Random Forest**.
3. **Production-Style Model Pipeline (`src/train.py`)**  
   Trains a Scikit-Learn `Pipeline` (`SimpleImputer` + `StandardScaler` + `OneHotEncoder` + `RandomForestClassifier`), reports **Accuracy, Precision, Recall, and F1-Score**, and saves `models/churn_model.pkl` with `joblib`.
4. **Interactive Streamlit Dashboard (`app.py`)**  
   - **Tab 1 (Data Explorer):** KPI cards, missing value report, interactive filters, and Matplotlib/Seaborn charts.
   - **Tab 2 (Churn Predictor):** Customer profile sliders/dropdowns that predict churn (`Yes`/`No`) with probability `%` and plain-English risk signals.

---

## 📁 Project Structure

```text
├── app.py                              # Streamlit dashboard (local port 8502)
├── vercel.json                         # Vercel static deployment config (serves web/)
├── web/
│   ├── index.html                      # Interactive web dashboard for Vercel
│   ├── dashboard_data.json             # Exported EDA stats & dataset for browser charts
│   └── rf_model.json                   # Exported Random Forest trees (exact sklearn parity)
├── data/
│   ├── generate_data.py                # Dataset generator & cleaning demo
│   ├── churn_data.csv                  # Generated raw dataset (5,000 rows)
│   └── churn_data_cleaned.csv          # Cleaned dataset after imputation
├── models/
│   └── churn_model.pkl                 # Trained Scikit-Learn Random Forest pipeline
├── notebooks/
│   └── EDA_and_Modeling.ipynb          # EDA, charts, & model comparison notebook
├── src/
│   ├── train.py                        # Model training & evaluation script
│   └── export_web_assets.py            # Exports dataset & trained RF model for web/
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Git ignore rules
└── README.md                           # Project documentation
```

---

## 🚀 Quick Start (Windows PowerShell)

```powershell
# 1. Allow local virtual environment activation
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force

# 2. Create and activate Python 3.12 virtual environment
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install --prefer-binary -r requirements.txt

# 4. Generate dataset & train the model
python data/generate_data.py
python src/train.py

# 5. Launch the Streamlit dashboard on port 8502
python -m streamlit run app.py --server.port 8502
```

Then open **http://localhost:8502** in your browser.

---

## 🌐 Deploying Live

- **Vercel (configured out of the box via `vercel.json`):**  
  Streamlit uses WebSockets and cannot run natively inside Vercel serverless functions, so this repo includes a matching web version in `web/` powered by the exact exported Random Forest trees (`src/export_web_assets.py`). Whenever you retrain `models/churn_model.pkl`, run `python src/export_web_assets.py` and push to `main` to update Vercel.
- **Streamlit Community Cloud:**  
  To host the Streamlit UI (`app.py`) directly, connect this GitHub repo at [share.streamlit.io](https://share.streamlit.io) with main file `app.py`.

---

## 💬 Interview Talking Points (For Junior Data Scientists)

- **Why use a Scikit-Learn `Pipeline` + `ColumnTransformer`?**  
  It prevents **data leakage** by fitting imputers, scalers, and encoders strictly on training folds, and bundles preprocessing + model into a single `.pkl` file for deployment.
- **Why Median for numeric imputation and Mode for categorical?**  
  Median is robust to skewed distributions/outliers (like high monthly bills), while mode preserves the most likely category without creating synthetic labels.
- **Why look beyond Accuracy?**  
  Churn datasets are imbalanced (~30% churn). Evaluating **Precision**, **Recall**, and **F1-Score** helps balance catching at-risk customers (Recall) against avoiding unnecessary retention offers (Precision).
