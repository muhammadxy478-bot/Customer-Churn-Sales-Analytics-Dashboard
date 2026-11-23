"""
Telecom Customer Churn Dataset Generator & Data Cleaning Script
---------------------------------------------------------------
This script:
1. Generates a realistic synthetic telecom dataset (5,000 rows) with
   realistic churn drivers (contract type, tenure, charges, support calls).
2. Injects ~3% missing values into select columns to mimic real-world data.
3. Saves the raw dataset to `data/churn_data.csv` (used by EDA, training,
   and the Streamlit Missing Value Report).
4. Demonstrates a clean, beginner-friendly data cleaning function and saves
   the cleaned version to `data/churn_data_cleaned.csv`.
"""

from pathlib import Path
import numpy as np
import pandas as pd


# Resolve paths relative to the project root so the script works from any directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "churn_data.csv"
CLEAN_DATA_PATH = DATA_DIR / "churn_data_cleaned.csv"


def generate_telecom_data(n_rows: int = 5000, random_seed: int = 42) -> pd.DataFrame:
    """Create a realistic synthetic telecom customer dataset with missing values."""
    rng = np.random.default_rng(random_seed)

    # 1. Customer IDs
    customer_ids = [f"CUST-{i:04d}" for i in range(1, n_rows + 1)]

    # 2. Contract Type (most customers are Month-to-Month)
    contract_type = rng.choice(
        ["Month-to-Month", "One Year", "Two Year"],
        size=n_rows,
        p=[0.55, 0.25, 0.20],
    )

    # 3. Internet Service
    internet_service = rng.choice(
        ["Fiber Optic", "DSL", "No"],
        size=n_rows,
        p=[0.45, 0.35, 0.20],
    )

    # 4. Tenure (in months): longer contracts tend to have longer tenure
    tenure = np.zeros(n_rows, dtype=int)
    for idx, contract in enumerate(contract_type):
        if contract == "Month-to-Month":
            tenure[idx] = rng.integers(1, 36)
        elif contract == "One Year":
            tenure[idx] = rng.integers(12, 60)
        else:
            tenure[idx] = rng.integers(24, 73)

    # 5. Monthly Charges ($): depends primarily on internet service tier
    monthly_charges = np.zeros(n_rows, dtype=float)
    for idx, service in enumerate(internet_service):
        if service == "Fiber Optic":
            monthly_charges[idx] = rng.normal(loc=90.0, scale=14.0)
        elif service == "DSL":
            monthly_charges[idx] = rng.normal(loc=58.0, scale=11.0)
        else:
            monthly_charges[idx] = rng.normal(loc=26.0, scale=5.0)
    monthly_charges = np.round(np.clip(monthly_charges, 20.0, 125.0), 2)

    # 6. Support Calls (in the past 6 months): Poisson distribution, slightly higher for Fiber
    base_calls = np.where(internet_service == "Fiber Optic", 2.4, 1.4)
    support_calls = np.clip(rng.poisson(lam=base_calls), 0, 10)

    # 7. Realistic Churn Probability using a logistic (sigmoid) formula
    # Positive weights increase churn risk; negative weights reduce churn risk.
    logit = (
        -2.2
        + np.where(contract_type == "Month-to-Month", 1.55, 0.0)
        + np.where(contract_type == "Two Year", -1.25, 0.0)
        + np.where(internet_service == "Fiber Optic", 0.65, 0.0)
        + np.where(internet_service == "No", -0.55, 0.0)
        - 0.045 * tenure
        + 0.018 * (monthly_charges - 60.0)
        + 0.42 * support_calls
        + rng.normal(0.0, 0.45, size=n_rows)  # random real-world noise
    )
    churn_prob = 1.0 / (1.0 + np.exp(-logit))
    churn_binary = rng.binomial(n=1, p=churn_prob)
    churn = np.where(churn_binary == 1, "Yes", "No")

    # Assemble DataFrame
    df = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "tenure": tenure,
            "monthly_charges": monthly_charges,
            "contract_type": contract_type,
            "internet_service": internet_service,
            "support_calls": support_calls,
            "churn": churn,
        }
    )

    # 8. Inject realistic missing values (~3% in three columns) for data cleaning demo
    for col, missing_frac in [
        ("monthly_charges", 0.03),
        ("internet_service", 0.025),
        ("support_calls", 0.02),
    ]:
        missing_indices = rng.choice(
            df.index, size=int(n_rows * missing_frac), replace=False
        )
        df.loc[missing_indices, col] = np.nan

    return df


def clean_telecom_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Demonstrate real-world data cleaning:
    - Remove duplicate customer records (if any).
    - Fill missing numerical columns with the median (robust to outliers).
    - Fill missing categorical columns with the mode (most frequent category).
    - Ensure integer columns have clean numeric types.
    """
    cleaned_df = df.copy()

    # Step 1: Drop exact duplicates if present
    cleaned_df = cleaned_df.drop_duplicates(subset=["customer_id"])

    # Step 2: Impute missing numeric values using the median
    numeric_cols = ["tenure", "monthly_charges", "support_calls"]
    for col in numeric_cols:
        if cleaned_df[col].isna().any():
            median_val = cleaned_df[col].median()
            cleaned_df[col] = cleaned_df[col].fillna(median_val)

    # Step 3: Impute missing categorical values using the mode
    categorical_cols = ["contract_type", "internet_service"]
    for col in categorical_cols:
        if cleaned_df[col].isna().any():
            mode_val = cleaned_df[col].mode()[0]
            cleaned_df[col] = cleaned_df[col].fillna(mode_val)

    # Step 4: Restore integer data types for count columns
    cleaned_df["tenure"] = cleaned_df["tenure"].round().astype(int)
    cleaned_df["support_calls"] = cleaned_df["support_calls"].round().astype(int)

    return cleaned_df


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Generate and save raw dataset (includes missing values for EDA & pipeline demo)
    raw_df = generate_telecom_data(n_rows=5000, random_seed=42)
    raw_df.to_csv(RAW_DATA_PATH, index=False)

    print(f"Generated raw dataset: {RAW_DATA_PATH} ({len(raw_df):,} rows)")
    print("\nMissing values BEFORE cleaning:")
    print(raw_df.isna().sum())

    # Run cleaning step and save cleaned dataset
    clean_df = clean_telecom_data(raw_df)
    clean_df.to_csv(CLEAN_DATA_PATH, index=False)

    print("\nMissing values AFTER cleaning:")
    print(clean_df.isna().sum())

    churn_rate = (clean_df["churn"] == "Yes").mean() * 100
    print(f"\nCleaned dataset saved: {CLEAN_DATA_PATH}")
    print(f"Overall Churn Rate: {churn_rate:.2f}%")


if __name__ == "__main__":
    main()
