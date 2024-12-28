import numpy as np
import pandas as pd
import os
import yaml

# Read the params.yaml file to get parameters
with open("params.yaml", 'r') as stream:
    params = yaml.safe_load(stream)

# Define raw data path
raw_data_path = "data_dvc/raw_data/MachineLearningRating_v3.csv"

# Define output directory and file
preprocessed_dir = "data_dvc/preprocessed"
preprocessed_file = os.path.join(preprocessed_dir, "MachineLearningRating_cleaned.csv")
os.makedirs(preprocessed_dir, exist_ok=True)

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
        print(f"Converting {column_name} to float type for numerical processing.")
        df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
    return df

def clean_column_cross_border(df):
    """
    Cleans the `CrossBorder` column by ensuring all entries are strings.
    """
    column_name = "CrossBorder"
    df[column_name] = df[column_name].fillna("").astype(str)
    print(f"{column_name} cleaned: now has types {df[column_name].apply(type).unique()}")
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

# Load raw data
df_raw = pd.read_csv(raw_data_path)

# Apply initial processing
df_initial = initial_processing(df_raw)

# Drop empty columns
df_no_empty_columns = drop_empty_column(df_initial)
