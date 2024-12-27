import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class EDA:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def summarize_data(self):
        """
        Summarize data by calculating descriptive statistics and reviewing data types.
        """
        print("Descriptive Statistics:")
        print(self.data.describe(include='all'))
        print("\nData Types:")
        print(self.data.dtypes)

    def assess_data_quality(self):
        """
        Check for missing values in the dataset.
        """
        print("Missing Values:")
        print(self.data.isnull().sum())

    def univariate_analysis(self):
        """
        Perform univariate analysis by plotting histograms and bar charts.
        """
        # Histograms for numerical columns
        numerical_cols = self.data.select_dtypes(include=['number']).columns
        self.data[numerical_cols].hist(bins=20, figsize=(14, 10), edgecolor='black')
        plt.suptitle('Histograms for Numerical Columns')
        plt.show()

        # Bar charts for categorical columns
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            self.data[col].value_counts().plot(kind='bar', figsize=(10, 5), title=f"Bar Chart for {col}")
            plt.ylabel('Count')
            plt.show()
