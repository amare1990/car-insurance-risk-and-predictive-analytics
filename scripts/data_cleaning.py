import os
import pandas as pd


class DataPreprocessor:
    def __init__(self, input_file="data/raw_data.csv", output_file="data/cleaned_data.csv"):
        """
        Initialize the preprocessor with file paths.

        Parameters:
        - input_file: str, Path to the input raw data file.
        - output_file: str, Path to save the cleaned/preprocessed data.
        """
        self.input_file = input_file
        self.output_file = output_file

    def load_data(self):
        """
        Load the dataset from the input file.
        """
        try:
            print(f"Loading data from {self.input_file}...")
            return pd.read_csv(self.input_file)
        except FileNotFoundError:
            print(f"Error: File {self.input_file} not found.")
            raise

    def save_data(self, df):
        """
        Save the preprocessed dataset to the output file.
        """
        print(f"Saving cleaned data to {self.output_file}...")
        df.to_csv(self.output_file, index=False)
        print("Data saved successfully.")

    def clean_column_capital_outstanding(self, df):
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

    def clean_column_cross_border(self, df):
        """
        Cleans the `CrossBorder` column by ensuring all entries are strings.
        """
        column_name = "CrossBorder"
        df[column_name] = df[column_name].fillna("").astype(str)
        print(f"{column_name} cleaned: now has types {df[column_name].apply(type).unique()}")
        return df

    def verify_and_clean_transaction_month(self, df):
        """
        Verifies and standardizes the `TransactionMonth` column.
        """
        column_name = "TransactionMonth"
        df[column_name] = pd.to_datetime(df[column_name], errors='coerce')
        invalid_dates = df[column_name].isnull().sum()
        print(f"Invalid dates found in {column_name}: {invalid_dates}")
        return df

    def initial_processing(self):
        """
        Perform the initial data cleaning and save the results.
        """
        # Load the data
        df = self.load_data()

        # Perform cleaning steps
        df = self.clean_column_capital_outstanding(df)
        df = self.clean_column_cross_border(df)
        df_cleaned_initial = self.verify_and_clean_transaction_month(df)

        # Save cleaned data
        self.save_data(df_cleaned_initial)

        # Track with Git and DVC
        self.track_with_dvc()

        return df_cleaned_initial


