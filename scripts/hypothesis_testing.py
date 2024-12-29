
import numpy as np
import pandas as pd


class ABHypothesisTesting:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the class with the dataset.
        :param data: A pandas DataFrame containing the dataset.
        """
        self.data = data

    def handle_missing_values(self):
        """
        Handle missing values in the dataset:
        - Drop columns with > 30% missing values.
        - Impute remaining missing values with mean (numerical) or mode (categorical).
        """
        # Drop columns with > 30% missing values
        missing_percent = self.data.isnull().mean()
        columns_to_drop = missing_percent[missing_percent > 0.3].index.tolist()
        self.data.drop(columns=columns_to_drop, axis=1, inplace=True)
        print(f"Dropped columns: {columns_to_drop}")

        # Impute remaining missing values
        for column in self.data.columns:
            if self.data[column].isnull().sum() > 0:
                if self.data[column].dtype in ['float64', 'int64']:
                    self.data[column].fillna(self.data[column].mean(), inplace=True)
                else:
                    self.data[column].fillna(self.data[column].mode()[0], inplace=True)
        print("Imputed remaining missing values.")

    def select_metrics(self, kpi: str):
        """
        Select the KPI to measure the impact of features.
        :param kpi: Key performance indicator column name.
        """
        if kpi not in self.data.columns:
            raise ValueError(f"{kpi} is not a valid column in the dataset.")
        self.kpi = kpi
        print(f"KPI selected: {self.kpi}")
