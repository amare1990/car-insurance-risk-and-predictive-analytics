import os
import numpy as np
import pandas as pd


class DataCleaning:
    def __init__(self, input_file="data/MachineLearningRating_v3.csv", output_file="data/final_cleaned_data.csv"):
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

    def customized_drop_duplicates(self, df):
        # Before dropping duplicates, get the initial number of rows and identify duplicates
        initial_row_count = df.shape[0]

        # Identify duplicate rows (including the first occurrence)
        duplicate_rows = df[df.duplicated(keep=False)]

        # Initialize a dictionary to track dropped rows for each first occurrence
        dropped_rows_dict = {}

        # Iterate over duplicate rows to find the first occurrence and the rows to be dropped
        for idx in duplicate_rows.index:
            # Get the first occurrence index (where the duplicate group starts)
            first_occurrence = df.duplicated(keep='first').idxmax()

            if first_occurrence not in dropped_rows_dict:
                dropped_rows_dict[first_occurrence] = []

            # Append the index of the current duplicate row to the dictionary (excluding the first occurrence)
            if idx != first_occurrence:
                dropped_rows_dict[first_occurrence].append(idx)

        # Drop duplicates, keeping the first occurrence
        df_cleaned = df.drop_duplicates(keep='first')

        # After dropping duplicates, get the updated number of rows
        final_row_count = df_cleaned.shape[0]

        # Calculate the number of rows dropped
        rows_dropped = initial_row_count - final_row_count

        # Print the appropriate messages
        if rows_dropped > 0:
            print(f"Dropped {rows_dropped} duplicate rows.")
            print(f"First occurrences with dropped rows: {dropped_rows_dict}")
            print(f"Total rows after dropping duplicates: {final_row_count}")
        else:
            print("No duplicates found.")

        return df_cleaned, dropped_rows_dict

    def clean_column_capital_outstanding(self, df):
        """
        Cleans the `CapitalOutstanding` column by converting it to a consistent data type.
        """
        column_name = "CapitalOutstanding"
        # float_count = df[column_name].apply(type).isin([float]).sum()
        # str_count = df[column_name].apply(type).isin([str]).sum()
        float_count = sum(isinstance(val, float) for val in df[column_name])
        str_count = sum(isinstance(val, str) for val in df[column_name])


        if float_count == 2 and str_count > 0:  # Only 2 float values, others are strings
            print(f"Converting {column_name} to string type for consistency.")
            df[column_name] = df[column_name].fillna("").astype(str)
        else:
            print(f"Converting {column_name} to float type for numerical processing.")
            df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
        return df

    def clean_column_cross_border(self, df):
        """
        Cleans the `CrossBorder` column by handling 'No' values as strings and filling empty/NaN values.
        """
        column_name = "CrossBorder"
        no_count = (df[column_name] == "No").sum()
        nan_count = df[column_name].isnull().sum()

        print(f"{column_name}: {no_count} 'No' values and {nan_count} missing/NaN values.")

        # Convert the column to string, filling NaN values with "Unknown" or other placeholder
        print(f"Converting {column_name} to string type and handling missing values.")
        df[column_name] = df[column_name].fillna("Unknown").astype(str)

        # Verify the types of the cleaned column
        print(f"{column_name} cleaned: now has types {df[column_name].apply(type).unique()} and example values {df[column_name].unique()[:5]}")

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

    # def initial_processing(self):
    #     """
    #     Perform the initial data cleaning and save the results.
    #     """
    #     # Load the data
    #     df = self.load_data()

    #     # Perform cleaning steps
    #     df = self.clean_column_capital_outstanding(df)
    #     df = self.clean_column_cross_border(df)
    #     df_cleaned_initial = self.verify_and_clean_transaction_month(df)

    #     # Save cleaned data
    #     self.save_data(df_cleaned_initial)

    #     return df_cleaned_initial

    def drop_empty_column(self, df):
        """
        Drops entirely empty columns and saves the result.
        """
        df_cleaned2 = df.copy()
        for col in df.columns:
            if df[col].isna().all():
                print(f"Column {col} is entirely empty, dropping it.")
                df_cleaned2.drop(columns=[col], inplace=True)

        return df_cleaned2

    def clean_dataframe_and_save(self, df, threshold=30):
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
        return df

    def process_pipeline(self):
        """
        Executes the full data cleaning pipeline and saves the final cleaned data.
        """
        # Step 1: Load data
        df = self.load_data()

        # Step 2: Apply cleaning steps
        df, _ = self.customized_drop_duplicates(df)
        df = self.clean_column_capital_outstanding(df)
        df = self.clean_column_cross_border(df)
        df = self.verify_and_clean_transaction_month(df)
        df = self.drop_empty_column(df)
        df = self.clean_dataframe_and_save(df)

        # Save the final cleaned DataFrame
        self.save_data(df)
        print(f"Final cleaned data saved to {self.output_file}.")
        return df

    def save_data(self, df):
        """
        Save the cleaned DataFrame to the output file.
        """
        df.to_csv(self.output_file, index=False)
