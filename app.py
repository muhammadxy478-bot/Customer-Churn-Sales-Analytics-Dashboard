"""
Telecom Customer Churn Prediction & EDA Dashboard
-------------------------------------------------
Interactive Streamlit application with two tabs:
- Tab 1 (Data Explorer): KPI summary, missing value report, and interactive
  Matplotlib/Seaborn visualizations.
- Tab 2 (Churn Predictor): Customer profile form powered by the trained
  Scikit-Learn Random Forest pipeline (`models/churn_model.pkl`).

Run with:
    streamlit run app.py --server.port 8502
"""

from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st


# Resolve paths relative to this script
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "churn_data.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.pkl"

# Configure global chart styling
sns.set_theme(style="whitegrid")


@st.cache_data
def load_dataset(path: Path) -> pd.DataFrame:
    """Load the raw telecom churn dataset from CSV."""
    return pd.read_csv(path)


@st.cache_resource
def load_model(path: Path):
    """Load the trained Scikit-Learn Random Forest pipeline."""
    return joblib.load(path)


def clean_for_charts(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values (median/mode) so charts display complete distributions."""
    cleaned = df.copy()
    for col in ["tenure", "monthly_charges", "support_calls"]:
        cleaned[col] = cleaned[col].fillna(cleaned[col].median())
    for col in ["contract_type", "internet_service"]:
        cleaned[col] = cleaned[col].fillna(cleaned[col].mode()[0])
    return cleaned


def build_missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Create a clean table summarizing missing values and how they are handled."""
    missing_counts = df.isna().sum()
    missing_pct = (missing_counts / len(df)) * 100

    strategy_map = {
        "customer_id": "Identifier (dropped before modeling)",
        "tenure": "Median imputation",
        "monthly_charges": "Median imputation",
        "contract_type": "Mode imputation",
        "internet_service": "Mode imputation",
        "support_calls": "Median imputation",
        "churn": "Target variable (Yes=1, No=0)",
    }

    report = pd.DataFrame(
        {
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Missing Count": missing_counts.values,
            "Missing (%)": missing_pct.round(2).values,
            "Handling Strategy": [strategy_map.get(c, "Standard") for c in df.columns],
        }
    )
    return report


def render_data_explorer_tab(raw_df: pd.DataFrame) -> None:
    """Render Tab 1: Dataset KPIs, Missing Value Report, and Interactive Charts."""
    st.header("📊 Telecom Data Explorer")
    st.write(
        "Explore customer demographics, data quality, and key drivers of telecom churn."
    )

    clean_df = clean_for_charts(raw_df)

    # Interactive filters for the charts
    st.subheader("Filter Data for Charts")
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        selected_contracts = st.multiselect(
            "Contract Type",
            options=sorted(clean_df["contract_type"].unique()),
            default=sorted(clean_df["contract_type"].unique()),
        )
    with filter_col2:
        selected_internet = st.multiselect(
            "Internet Service",
            options=sorted(clean_df["internet_service"].unique()),
            default=sorted(clean_df["internet_service"].unique()),
        )

    filtered_df = clean_df[
        clean_df["contract_type"].isin(selected_contracts)
        & clean_df["internet_service"].isin(selected_internet)
    ]

    if filtered_df.empty:
        st.warning("No customers match the selected filters. Please select at least one option.")
        return

    # KPI Summary Metrics
    total_customers = len(filtered_df)
    churn_rate = (filtered_df["churn"] == "Yes").mean() * 100
    avg_monthly = filtered_df["monthly_charges"].mean()
    avg_tenure = filtered_df["tenure"].mean()
    total_missing = int(raw_df.isna().sum().sum())

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Customers Shown", f"{total_customers:,}")
    kpi2.metric("Churn Rate", f"{churn_rate:.1f}%")
    kpi3.metric("Avg Monthly Charge", f"${avg_monthly:.2f}")
    kpi4.metric("Avg Tenure", f"{avg_tenure:.1f} mos")
    kpi5.metric("Raw Missing Cells", f"{total_missing:,}")

    st.divider()

    # Missing Value Report (based on raw CSV)
    st.subheader("🧹 Data Quality & Missing Value Report (Raw Dataset)")
    missing_report = build_missing_value_report(raw_df)
    st.dataframe(missing_report, width="stretch", hide_index=True)

    st.divider()

    # Row 1 of Charts: Churn Count & Churn Rate by Contract Type
    st.subheader("📈 Visual Churn Analysis")
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(
            data=filtered_df,
            x="churn",
            hue="churn",
            palette={"No": "#2ecc71", "Yes": "#e74c3c"},
            legend=False,
            ax=ax,
        )
        ax.set_title("Customer Churn Distribution", fontsize=12, fontweight="bold")
        ax.set_xlabel("Churn")
        ax.set_ylabel("Number of Customers")
        for container in ax.containers:
            ax.bar_label(container, fmt="%d", padding=3)
        st.pyplot(fig)
        plt.close(fig)

    with chart_col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        contract_churn = (
            filtered_df.assign(churn_flag=(filtered_df["churn"] == "Yes").astype(int))
            .groupby("contract_type", as_index=False)["churn_flag"]
            .mean()
        )
        contract_churn["churn_pct"] = contract_churn["churn_flag"] * 100
        sns.barplot(
            data=contract_churn,
            x="contract_type",
            y="churn_pct",
            hue="contract_type",
            palette="viridis",
            legend=False,
            ax=ax,
        )
        ax.set_title("Churn Rate (%) by Contract Type", fontsize=12, fontweight="bold")
        ax.set_xlabel("Contract Type")
        ax.set_ylabel("Churn Rate (%)")
        ax.set_ylim(0, max(contract_churn["churn_pct"].max() * 1.2, 10))
        for container in ax.containers:
            ax.bar_label(container, fmt="%.1f%%", padding=3)
        st.pyplot(fig)
        plt.close(fig)

    # Row 2 of Charts: Monthly Charges by Churn & Support Calls by Churn
    chart_col3, chart_col4 = st.columns(2)

    with chart_col3:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(
            data=filtered_df,
            x="monthly_charges",
            hue="churn",
            bins=25,
            kde=True,
            palette={"No": "#2ecc71", "Yes": "#e74c3c"},
            alpha=0.55,
            ax=ax,
        )
        ax.set_title("Monthly Charges Distribution by Churn", fontsize=12, fontweight="bold")
        ax.set_xlabel("Monthly Charges ($)")
        ax.set_ylabel("Customer Count")
        st.pyplot(fig)
        plt.close(fig)

    with chart_col4:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(
            data=filtered_df,
            x="churn",
            y="support_calls",
            hue="churn",
            palette={"No": "#2ecc71", "Yes": "#e74c3c"},
            legend=False,
            ax=ax,
        )
        ax.set_title("Support Calls vs. Churn", fontsize=12, fontweight="bold")
        ax.set_xlabel("Churn")
        ax.set_ylabel("Support Calls (Past 6 Months)")
        st.pyplot(fig)
        plt.close(fig)

    # Row 3 of Charts: Tenure vs. Churn & Correlation Heatmap
    chart_col5, chart_col6 = st.columns(2)

    with chart_col5:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(
            data=filtered_df,
            x="churn",
            y="tenure",
            hue="churn",
            palette={"No": "#2ecc71", "Yes": "#e74c3c"},
            legend=False,
            ax=ax,
        )
        ax.set_title("Customer Tenure vs. Churn", fontsize=12, fontweight="bold")
        ax.set_xlabel("Churn")
        ax.set_ylabel("Tenure (Months)")
        st.pyplot(fig)
        plt.close(fig)

    with chart_col6:
        fig, ax = plt.subplots(figsize=(6, 4))
        corr_df = filtered_df[["tenure", "monthly_charges", "support_calls"]].copy()
        corr_df["churn_binary"] = (filtered_df["churn"] == "Yes").astype(int)
        corr_matrix = corr_df.corr()
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            square=True,
            ax=ax,
        )
        ax.set_title("Feature Correlation Heatmap", fontsize=12, fontweight="bold")
        st.pyplot(fig)
        plt.close(fig)

    # Optional preview of raw records
    with st.expander("Preview Raw Dataset (First 50 Rows)"):
        st.dataframe(raw_df.head(50), width="stretch")


def render_churn_predictor_tab(model) -> None:
    """Render Tab 2: Interactive Customer Churn Prediction Form."""
    st.header("🔮 Customer Churn Predictor")
    st.write(
        "Adjust the customer profile below to predict churn likelihood using the "
        "trained **Scikit-Learn Random Forest** pipeline."
    )

    input_col, result_col = st.columns([1.1, 1.0], gap="large")

    with input_col:
        st.subheader("Customer Profile Inputs")
        tenure = st.slider(
            "Tenure (Months with Company)",
            min_value=1,
            max_value=72,
            value=12,
            step=1,
        )
        monthly_charges = st.slider(
            "Monthly Charges ($)",
            min_value=20.0,
            max_value=125.0,
            value=75.0,
            step=0.5,
        )
        support_calls = st.slider(
            "Customer Support Calls (Past 6 Months)",
            min_value=0,
            max_value=10,
            value=2,
            step=1,
        )
        contract_type = st.selectbox(
            "Contract Type",
            options=["Month-to-Month", "One Year", "Two Year"],
            index=0,
        )
        internet_service = st.selectbox(
            "Internet Service",
            options=["Fiber Optic", "DSL", "No"],
            index=0,
        )

    # Assemble input into a single-row DataFrame matching training columns
    customer_df = pd.DataFrame(
        [
            {
                "tenure": tenure,
                "monthly_charges": monthly_charges,
                "support_calls": support_calls,
                "contract_type": contract_type,
                "internet_service": internet_service,
            }
        ]
    )

    # Generate prediction and probabilities
    churn_prob = float(model.predict_proba(customer_df)[0][1])
    retain_prob = 1.0 - churn_prob
    will_churn = churn_prob >= 0.50

    with result_col:
        st.subheader("Prediction Result")

        if will_churn:
            st.error(
                f"### ⚠️ High Churn Risk — Predicted: **YES**\n"
                f"Estimated Churn Probability: **{churn_prob * 100:.1f}%**"
            )
        else:
            st.success(
                f"### ✅ Likely to Stay — Predicted: **NO**\n"
                f"Estimated Churn Probability: **{churn_prob * 100:.1f}%**"
            )

        m1, m2 = st.columns(2)
        m1.metric("Churn Probability", f"{churn_prob * 100:.1f}%")
        m2.metric("Retention Probability", f"{retain_prob * 100:.1f}%")

        # Horizontal probability bar chart
        fig, ax = plt.subplots(figsize=(6, 2.2))
        categories = ["Stay (No)", "Churn (Yes)"]
        probs = [retain_prob * 100, churn_prob * 100]
        colors = ["#2ecc71", "#e74c3c"]
        bars = ax.barh(categories, probs, color=colors, height=0.5)
        ax.set_xlim(0, 105)
        ax.set_xlabel("Probability (%)")
        ax.set_title("Predicted Class Probabilities", fontsize=11, fontweight="bold")
        for bar, pct in zip(bars, probs):
            ax.text(
                pct + 1.5,
                bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%",
                va="center",
                fontweight="bold",
            )
        st.pyplot(fig)
        plt.close(fig)

        # Plain-English explanation of risk drivers for business clarity
        st.markdown("#### Key Profile Signals")
        signals = []
        if contract_type == "Month-to-Month":
            signals.append("🔴 **Month-to-Month contract** carries higher churn risk than annual plans.")
        else:
            signals.append(f"🟢 **{contract_type} contract** helps stabilize customer retention.")

        if support_calls >= 4:
            signals.append(f"🔴 **High support volume ({support_calls} calls)** often indicates unresolved issues.")
        elif support_calls <= 1:
            signals.append(f"🟢 **Low support calls ({support_calls})** suggests a smooth service experience.")

        if tenure < 12:
            signals.append(f"🟡 **Short tenure ({tenure} months)**: newer customers are more likely to switch.")
        elif tenure >= 36:
            signals.append(f"🟢 **Long tenure ({tenure} months)** signals strong loyalty.")

        if internet_service == "Fiber Optic" and monthly_charges >= 85:
            signals.append("🟡 **High Fiber Optic bill** may increase price sensitivity.")

        for note in signals:
            st.markdown(f"- {note}")


def main() -> None:
    st.set_page_config(
        page_title="Telecom Customer Churn Dashboard",
        page_icon="📡",
        layout="wide",
    )

    st.title("📡 Telecom Customer Churn Prediction & EDA Dashboard")
    st.caption(
        "Built with Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn, and Streamlit."
    )

    # Verify required data and model artifacts exist
    if not DATA_PATH.exists():
        st.error(
            f"Dataset not found at `{DATA_PATH}`. Run `python data/generate_data.py` first."
        )
        st.stop()

    if not MODEL_PATH.exists():
        st.error(
            f"Trained model not found at `{MODEL_PATH}`. Run `python src/train.py` first."
        )
        st.stop()

    raw_df = load_dataset(DATA_PATH)
    model = load_model(MODEL_PATH)

    tab1, tab2 = st.tabs(["📊 Data Explorer", "🔮 Churn Predictor"])

    with tab1:
        render_data_explorer_tab(raw_df)

    with tab2:
        render_churn_predictor_tab(model)


if __name__ == "__main__":
    main()
