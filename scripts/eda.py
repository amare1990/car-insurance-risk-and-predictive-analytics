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

    def bivariate_analysis(self):
        """
        Explore relationships between numerical variables using scatter plots and correlation matrices.
        """
        # Correlation Matrix
        correlation_matrix = self.data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix')
        plt.show()

        # Scatter Plot for TotalPremium vs TotalClaims by ZipCode
        if {'TotalPremium', 'TotalClaims', 'ZipCode'}.issubset(self.data.columns):
            plt.figure(figsize=(10, 6))
            sns.scatterplot(
                data=self.data, x='TotalPremium', y='TotalClaims', hue='ZipCode', palette='viridis'
            )
            plt.title('TotalPremium vs TotalClaims by ZipCode')
            plt.show()

    def data_comparison(self):
        """
        Compare trends over geography and other categorical factors.
        """
        if 'PostalCode' in self.data.columns:
            grouped = self.data.groupby('PostalCode').mean()
            grouped[['TotalPremium', 'TotalClaims']].plot(kind='bar', figsize=(12, 6), title='Average Premium and Claims by ZipCode')
            plt.ylabel('Average Value')
            plt.show()
