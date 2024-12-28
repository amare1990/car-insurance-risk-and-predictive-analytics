import numpy as np
import pandas as pd
import os
import yaml

# Read the params.yaml file to get parameters
with open("params.yaml", 'r') as stream:
    params = yaml.safe_load(stream)

# Read raw data
# raw_data_path = "data_dvc/raw_data/MachineLearningRating_v3.csv"
# df = pd.read_csv(raw_data_path)

# def clean_column_capital_outstanding(df):
#     """
#     Cleans the `CapitalOutstanding` column by converting it to a consistent data type.
#     - Converts to `str` if most values are strings.
#     - Converts to `float` if numeric data is appropriate.
#     """
#     column_name = "CapitalOutstanding"
#     float_count = df[column_name].apply(type).isin([float]).sum()
#     str_count = df[column_name].apply(type).isin([str]).sum()

#     if float_count == 2 and str_count > 0:  # Only 2 float values, others are strings
#         print(f"Converting {column_name} to string type for consistency.")
#         df[column_name] = df[column_name].fillna("").astype(str)
#     else:
#         print(f"Converting {column_name} to float type for numerical processing.")
#         df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
#     return df

# def clean_column_cross_border(df):
#     """
#     Cleans the `CrossBorder` column by ensuring all entries are strings.
#     - Replaces 'No' with 'No' (not encoding).
#     - Ensures the column is of string type.
#     """
#     column_name = "CrossBorder"
#     # Retain 'No' and replace NaN/float values with empty string
#     df[column_name] = df[column_name].fillna("").astype(str)
#     print(f"{column_name} cleaned: now has types {df[column_name].apply(type).unique()}")
#     return df

# def verify_and_clean_transaction_month(df):
#     """
#     Verifies and standardizes the transaction date column.
#     - Converts non-standard formats to NaT.
#     - Ensures consistency in date format.
#     """
#     column_name = "TransactionMonth"
#     df[column_name] = pd.to_datetime(df[column_name], errors='coerce')
#     invalid_dates = df[column_name].isnull().sum()
#     print(f"Invalid dates found in {column_name}: {invalid_dates}")
#     return df


# Load raw data
# df = pd.read_csv(raw_data_path)

# Fixing mixed data types and checking date time format
# def initial_processing(df):
#     df = clean_column_capital_outstanding(df)
#     df = clean_column_cross_border(df)
#     df_cleaned_initial= verify_and_clean_transaction_month(df)

#     return df_cleaned_initial

# # Dropping columns which are entirely empty
# df_cleaned_initial = initial_processing(df)

# Dropping columns which are entirely empty
preprocessed_data_path = "data_dvc/preprocessed/MachineLearningRating_v3.csv"
preprocessed_data = pd.read_csv(preprocessed_data_path)
def drop_empty_column(df_cleaned1):
    """
    Drops entirely empty columns from the DataFrame and returns the cleaned DataFrame.
    """
    df_cleaned2 = preprocessed_data.copy()
    for col in preprocessed_data.columns:
        if preprocessed_data[col].isna().all():
            print(f"Column {col} is entirely empty, dropping the column ongoing!")
            df_cleaned2.drop(columns=[col], inplace=True)
    return df_cleaned2

# df_cleaned_initial = initial_processing(df)



# After preprocessing
output_dir = "data_dvc/preprocessed"
os.makedirs(output_dir, exist_ok=True)

df_empty_col = drop_empty_column(preprocessed_data)
preprocessed_file = os.path.join(output_dir, "MachineLearningRating_cleaned.csv")
df_empty_col.to_csv(preprocessed_file, index=False)

# preprocessed_file = os.path.join(output_dir, "MachineLearningRating_cleaned.csv")
# df_empty_col.to_csv(preprocessed_file, index=False)


# Check if the file exists
if os.path.exists(preprocessed_file):
    print(f"Preprocessed data saved to {preprocessed_file}")
else:
    print(f"Error: Preprocessed data file not found: {preprocessed_file}")



# # Add and track the changes with Git and DVC
# os.system(f"git add {preprocessed_file}")  # Add to Git
# os.system(f"dvc add {preprocessed_file}")  # Add to DVC
# os.system("git commit -m 'Overwrite and add cleaned dataset version'")  # Commit to Git
# os.system("dvc push")  # Push the DVC data to the remote storage

print(f"Preprocessed data saved to {preprocessed_file}")
