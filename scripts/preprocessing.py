"""
A data preprocessing script with many pipeline stages
"""
import os
import numpy as np
import pandas as pd
import yaml

# Locate params.yaml relative to the script's location
# Directory of the preprocessing.py script
script_dir = os.path.dirname(__file__)
params_path = os.path.join(script_dir, "../params.yaml")  # Adjust as needed


# Read the params.yaml file to get parameters
with open(params_path, 'r', encoding='utf-8') as stream:
    params = yaml.safe_load(stream)

# Define raw data path
RAW_DATA_PATH = "data_dvc/raw_data/MachineLearningRating_v3.csv"

# Define output directory and file
PREPROCEESING_DIR = "data_dvc/preprocessed"
preprocessed_file = os.path.join(
    PREPROCEESING_DIR,
    "MachineLearningRating_cleaned.csv")
os.makedirs(PREPROCEESING_DIR, exist_ok=True)


def clean_column_capital_outstanding(df):
    """
    Cleans the `CapitalOutstanding` column by converting it to a consistent data type.
    """
    column_name = "CapitalOutstanding"
    float_count = df[column_name].apply(type).isin([float]).sum()
    str_count = df[column_name].apply(type).isin([str]).sum()

    if float_count == 2 and str_count > 0:  # Only 2 float values, others are strings
        print(f"Converting {column_name} to string type for consistency.")
        df[column_name] = df[column_name].fillna("").astype(str)
    else:
        print(
            f"Converting {column_name} to float type for numerical processing.")
        df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
    return df


def clean_column_cross_border(df):
    """
    Cleans the `CrossBorder` column by ensuring all entries are strings.
    """
    column_name = "CrossBorder"
    df[column_name] = df[column_name].fillna("").astype(str)
    print(
        f"{column_name} cleaned: now has types {df[column_name].apply(type).unique()}")
    return df


def verify_and_clean_transaction_month(df):
    """
    Verifies and standardizes the transaction date column.
    """
    column_name = "TransactionMonth"
    df[column_name] = pd.to_datetime(df[column_name], errors='coerce')
    invalid_dates = df[column_name].isnull().sum()
    print(f"Invalid dates found in {column_name}: {invalid_dates}")
    return df


def initial_processing(df):
    """
    Performs the initial data cleaning and saves the result.
    """
    df = clean_column_capital_outstanding(df)
    df = clean_column_cross_border(df)
    df_cleaned_initial = verify_and_clean_transaction_month(df)

    # Save the result
    df_cleaned_initial.to_csv(preprocessed_file, index=False)

    # Track with Git and DVC
    os.system(f"git add {preprocessed_file}")
    os.system(f"dvc add {preprocessed_file}")
    os.system("git commit -m 'Initial processing of data'")
    os.system("dvc push")

    return df_cleaned_initial


def drop_empty_column(df):
    """
    Drops entirely empty columns and saves the result.
    """
    df_cleaned2 = df.copy()
    for col in df.columns:
        if df[col].isna().all():
            print(f"Column {col} is entirely empty, dropping it.")
            df_cleaned2.drop(columns=[col], inplace=True)

    # Save the result
    df_cleaned2.to_csv(preprocessed_file, index=False)

    # Track with Git and DVC
    os.system(f"git add {preprocessed_file}")
    os.system(f"dvc add {preprocessed_file}")
    os.system("git commit -m 'Dropped empty columns from data'")
    os.system("dvc push")

    return df_cleaned2


def clean_dataframe_and_save(df, threshold=30):
    """
    Cleans the DataFrame by handling missing/null/infinity values based on the threshold percentage.
    - Drops columns with more than `threshold`% missing values.
    - Replaces numeric columns with their mean if missing values are less than or equal
      to `threshold`%.
    - Replaces categorical columns with their mode if missing values are less than or equal to
      `threshold`%.
    Saves the cleaned DataFrame and tracks it with Git and DVC.
    """
    total_rows = df.shape[0]
    # Calculate the drop threshold
    drop_threshold = (threshold / 100) * total_rows

    for col in df.columns:
        # Handle missing values
        missing_count = df[col].isnull().sum()

        # Handle infinite values only for numeric columns
        if df[col].dtype in [np.float64, np.int64]:
            missing_count += np.isinf(df[col].replace(
                [np.inf, -np.inf], np.nan)).sum()

        if missing_count > drop_threshold:
            # Drop column if missing values exceed threshold
            print(
                f"Dropping column {col} with {missing_count} missing/infinite values.")
            df.drop(columns=[col], inplace=True)
        elif missing_count > 0:  # Only process columns with missing values
            if np.issubdtype(df[col].dtype, np.number):
                # Replace numeric columns with mean
                print(
                    f"Replacing missing values in numeric column {col} with mean.")
                df[col].replace([np.inf, -np.inf], np.nan, inplace=True)
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                # Replace categorical columns with mode
                print(
                    f"Replacing missing values in categorical column {col} with mode.")
                df[col].fillna(df[col].mode()[0], inplace=True)

    # Save the cleaned DataFrame
    df_cleaned_path = os.path.join(
        PREPROCEESING_DIR,
        "MachineLearningRating_cleaned_final.csv")
    df.to_csv(df_cleaned_path, index=False)

    # Track with Git and DVC
    os.system(f"git add {df_cleaned_path}")
    os.system(f"dvc add {df_cleaned_path}")
    os.system(
        "git commit -m 'Cleaned DataFrame by handling missing/infinite values'")
    os.system("dvc push")

    return df


# Load raw data
df_raw = pd.read_csv(RAW_DATA_PATH, low_memory=False)

# Apply initial processing
df_initial = initial_processing(df_raw)

# Drop empty columns
df_no_empty_columns = drop_empty_column(df_initial)

# Incorporate into the pipeline
# Apply data cleaning
df_cleaned = clean_dataframe_and_save(
    df_no_empty_columns,
    threshold=params["cleaning"]["threshold"])
