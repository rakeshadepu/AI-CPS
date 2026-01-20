import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")


# ======================================================
# DATA LOADING & CLEANING
# ======================================================
def load_and_clean_raw_data():
    """
    Load raw scraped data and perform initial cleaning.
    """

    df = pd.read_csv("../data/raw/scraped_data.csv")

    # Remove unwanted row
    df = df.drop(index=16)

    # Fix encoding issue
    df["state"] = df["state"].str.replace(
        "Baden-WÃ¼rttemberg",
        "Baden-Wurttemberg",
        regex=False
    )

    # Drop derived share columns
    share_cols = [col for col in df.columns if "Share" in col]
    if share_cols:
        df = df.drop(columns=share_cols)

    return df


# ======================================================
# DATA TRANSFORMATION
# ======================================================
def transform_wide_to_long(df):
    """
    Transform wide-format data into long-format
    for time-series analysis.
    """

    pop_cols = [c for c in df.columns if c.startswith("Population_")]
    emp_cols = [c for c in df.columns if c.startswith("Employed")]
    gdp_cols = [c for c in df.columns if c.startswith("Gross")]

    def melt_block(value_cols, value_name):
        """
        Helper function to melt a group of columns.
        """
        melted = df.melt(
            id_vars=["state"],
            value_vars=value_cols,
            var_name="year",
            value_name=value_name
        )
        melted["year"] = melted["year"].str.extract(r"(\d{4})").astype(int)
        return melted

    pop_long = melt_block(pop_cols, "population")
    emp_long = melt_block(emp_cols, "employment")
    gdp_long = melt_block(gdp_cols, "gdp")

    long_df = (
        pop_long
        .merge(emp_long, on=["state", "year"])
        .merge(gdp_long, on=["state", "year"])
        .sort_values(["state", "year"])
        .reset_index(drop=True)
    )

    return long_df


# ======================================================
# MISSING VALUES
# ======================================================
def handle_missing_values(df):
    """
    Detect and handle missing values.
    """

    df = df.replace(["NA", "N/A", "null", "-", "", " "], np.nan)

    missing = df.isnull().sum()
    total_missing = missing.sum()

    for col, count in missing.items():
        if count > 0:
            pct = (count / len(df)) * 100
            print(f"  {col}: {count} ({pct:.1f}%)")

    if total_missing > 0:
        initial_len = len(df)
        df = df.dropna()
        print(f"Dropped {initial_len - len(df)} rows with missing values")
        print(f"Remaining rows: {len(df)}")
    else:
        print("No missing values found")

    return df


# ======================================================
# OUTLIER ANALYSIS (NO REMOVAL)
# ======================================================
def analyze_outliers_without_removal(df, columns):
    """
    Analyze outliers using IQR method WITHOUT removing them.
    """

    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers = (df[col] < lower) | (df[col] > upper)
        n_outliers = outliers.sum()

        print(f"\n{col}:")
        print(f"  Range: [{df[col].min():,.0f}, {df[col].max():,.0f}]")
        print(f"  IQR bounds: [{lower:,.0f}, {upper:,.0f}]")
        print(f"  Outliers: {n_outliers} ({n_outliers / len(df) * 100:.1f}%)")

        if n_outliers > 0:
            states = df.loc[outliers, "state"].unique()
            print(f"  States flagged: {', '.join(states)}")

    print("\n" + "=" * 60)
    print("DECISION: Outliers NOT removed")
    print("=" * 60)

    return df


# ======================================================
# LOG TRANSFORMATION
# ======================================================
def validate_data_for_log_transform(df, columns):
    """
    Validate data for log transformation.
    """

    for col in columns:
        non_positive = (df[col] <= 0).sum()
        if non_positive > 0:
            print(f"⚠️  {col} has {non_positive} non-positive values")
            df = df[df[col] > 0]

    return df


def apply_log_transformation(df, columns):
    """
    Apply logarithmic transformation.
    """

    for col in columns:
        df[col] = np.log(df[col])
        print(f"{col}: log transformation applied")

    return df


# ======================================================
# NORMALIZATION
# ======================================================
def apply_normalization(df, columns):
    """
    Normalize data using StandardScaler.
    """

    scaler = StandardScaler()

    print("Original statistics:")
    for col in columns:
        print(f"  {col}: mean={df[col].mean():.4f}, std={df[col].std():.4f}")

    df[columns] = scaler.fit_transform(df[columns])

    print("\nNormalized statistics:")
    for col in columns:
        print(f"  {col}: mean={df[col].mean():.4f}, std={df[col].std():.4f}")

    return df


# ======================================================
# DATA SPLITTING & SAVING
# ======================================================
def create_train_test_split(df, test_size=0.2, random_state=42):
    """
    Split data into training and test sets.
    """

    return train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )


def create_activation_sample(test_df, random_state=42):
    """
    Create a single activation sample.
    """

    activation_df = test_df.sample(n=1, random_state=random_state)
    print("Activation sample created")
    return activation_df


def save_datasets(joint_df, train_df, test_df, activation_df):
    """
    Save processed datasets.
    """

    output_dir = Path("../data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    joint_df.to_csv(output_dir / "joint_data_collection.csv", index=False)
    train_df.to_csv(output_dir / "training_data.csv", index=False)
    test_df.to_csv(output_dir / "test_data.csv", index=False)
    activation_df.to_csv(output_dir / "activation_data.csv", index=False)

    print(f"\nAll files saved to: {output_dir.resolve()}")


# ======================================================
# MAIN PIPELINE
# ======================================================
def main():
    """
    Execute full data preparation pipeline.
    """

    df = load_and_clean_raw_data()
    long_df = transform_wide_to_long(df)
    long_df = handle_missing_values(long_df)

    numeric_cols = ["population", "employment", "gdp"]
    long_df = analyze_outliers_without_removal(long_df, numeric_cols)

    joint_df_clean = long_df.copy()

    long_df = validate_data_for_log_transform(long_df, numeric_cols)
    long_df = apply_log_transformation(long_df, numeric_cols)
    long_df = apply_normalization(long_df, numeric_cols)

    train_df, test_df = create_train_test_split(long_df)
    activation_df = create_activation_sample(test_df)

    save_datasets(joint_df_clean, train_df, test_df, activation_df)

    print("\n=== FINAL SUMMARY ===")
    print(f"States: {train_df['state'].nunique()}")
    print(f"Features: {list(train_df.columns)}")
    print(f"Training rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"Activation rows: {len(activation_df)}")


if __name__ == "__main__":
    main()
