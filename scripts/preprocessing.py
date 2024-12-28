import numpy as np
import pandas as pd
import os
import yaml

# Read the params.yaml file to get parameters
with open("params.yaml", 'r') as stream:
    params = yaml.safe_load(stream)

# Read raw data
raw_data_path = "data_dvc/raw_data/MachineLearningRating_v3.csv"
df = pd.read_csv(raw_data_path)

def clean_column_capital_outstanding(df):
    """
    Cleans the `CapitalOutstanding` column by converting it to a consistent data type.
    - Converts to `str` if most values are strings.
    - Converts to `float` if numeric data is appropriate.
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
    - Replaces 'No' with 'No' (not encoding).
    - Ensures the column is of string type.
    """
    column_name = "CrossBorder"
    # Retain 'No' and replace NaN/float values with empty string
    df[column_name] = df[column_name].fillna("").astype(str)
    print(f"{column_name} cleaned: now has types {df[column_name].apply(type).unique()}")
    return df

def verify_and_clean_transaction_month(df):
    """
    Verifies and standardizes the transaction date column.
    - Converts non-standard formats to NaT.
    - Ensures consistency in date format.
    """
    column_name = "TransactionMonth"
    df[column_name] = pd.to_datetime(df[column_name], errors='coerce')
    invalid_dates = df[column_name].isnull().sum()
    print(f"Invalid dates found in {column_name}: {invalid_dates}")
    return df


# Load raw data
df = pd.read_csv(raw_data_path)

# Clean specific columns
df = clean_column_capital_outstanding(df)
df = clean_column_cross_border(df)
df = verify_and_clean_transaction_month(df)