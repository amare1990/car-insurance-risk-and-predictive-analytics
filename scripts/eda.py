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

    def geographical_trends(self):
        """Compares trends over geography for features like CoverType, make, etc."""

        if {"Province", "CoverType", "make"}.issubset(self.data.columns):
            for col in ["CoverType", "make"]:
                plt.figure(figsize=(10, 6))
                sns.countplot(data=self.data, y=col, hue="Province", order=self.data[col].value_counts().index)
                plt.title(f"Geographical Trends of {col}")
                plt.show()
        else:
            print("Required columns for geographical trends are not available.")

    def detect_outliers(self):
        """
        Use box plots to detect outliers in numerical data.
        """
        numerical_cols = self.data.select_dtypes(include=['number']).columns
        for col in numerical_cols:
            plt.figure(figsize=(10, 5))
            sns.boxplot(data=self.data, x=col)
            plt.title(f"Box Plot for {col}")
            plt.show()

    def create_insightful_visualizations(self):
        """Creates insightful visualizations."""
        if {"TotalPremium", "TotalClaims", "VehicleType"}.issubset(self.data.columns):
            # Visualization 1: Premium vs. Claims by Vehicle Type
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=self.data, x="VehicleType", y="TotalPremium")
            plt.title("Total Premium Distribution by Vehicle Type")
            plt.show()

            # Visualization 2: Claims Distribution by CoverType
            plt.figure(figsize=(10, 6))
            sns.barplot(data=self.data, x="CoverType", y="TotalClaims", estimator=sum, ci=None)
            plt.title("Total Claims by CoverType")
            plt.show()

            # Visualization 3: Premiums by Province
            plt.figure(figsize=(10, 6))
            sns.barplot(data=self.data, x="Province", y="TotalPremium", estimator=sum, ci=None)
            plt.title("Total Premiums by Province")
            plt.xticks(rotation=45)
            plt.show()
        else:
            print("Columns for insightful visualizations are not available in the dataset.")
