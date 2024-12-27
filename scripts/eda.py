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
        print("\nUnique Values in Categorical Columns:\n")
        for col in self.data.select_dtypes(include=["object", "category"]):
            print(f"{col}: {self.data[col].nunique()} unique values")

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
        plt.savefig(f"plots/histogram.png", dpi=300, bbox_inches='tight')
        plt.show()

        # Bar charts for categorical columns
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            self.data[col].value_counts().plot(kind='bar', figsize=(10, 5), title=f"Bar Chart for {col}")
            plt.ylabel('Count')
            plt.savefig(f"plots/barchart/Bar Chart for {col}.png", dpi=300, bbox_inches='tight')
            plt.show()

    def bivariate_analysis(self):
        """Explores relationships between features like TotalPremium, TotalClaims, and PostalCode."""
        if {"TotalPremium", "TotalClaims", "PostalCode"}.issubset(self.data.columns):
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=self.data, x="TotalPremium", y="TotalClaims", hue="PostalCode")
            plt.title("Scatter Plot of TotalPremium vs. TotalClaims by PostalCode")
            plt.savefig('plots/scatter plotst.png', dpi=300, bbox_inches='tight')
            plt.show()

            correlation_matrix = self.data[["TotalPremium", "TotalClaims"]].corr()
            print("Correlation Matrix:\n", correlation_matrix)
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
            plt.title("Correlation Matrix Heatmap")
            plt.savefig('plots/correlation matrix.png', dpi=300, bbox_inches='tight')
            plt.show()
        else:
            print("Columns for bivariate analysis are not available in the dataset.")

    def geographical_trends(self):
        """Compares trends over geography for features like CoverType, make, etc."""

        if {"Province", "CoverType", "make"}.issubset(self.data.columns):
            for col in ["CoverType", "make"]:
                plt.figure(figsize=(10, 6))
                sns.countplot(data=self.data, y=col, hue="Province", order=self.data[col].value_counts().index)
                plt.title(f"Geographical Trends of {col}")
                plt.savefig(f"plots/countplot for {col}.png", dpi=300, bbox_inches='tight')
                plt.show()
        else:
            print("Required columns for geographical trends are not available.")

    def detect_outliers(self):
      """
      Use box plots to detect outliers in numerical data.
      Handles missing or invalid values in the dataset.
      """
      numerical_cols = self.data.select_dtypes(include=['number']).columns

      for col in numerical_cols:
          # Drop rows with NaN or infinite values in the current column
          valid_data = self.data[col].dropna()
          valid_data = valid_data[np.isfinite(valid_data)]

          if valid_data.nunique() > 1:
              plt.figure(figsize=(10, 5))
              sns.boxplot(x=valid_data)
              plt.title(f"Box Plot for {col}")
              plt.savefig(f"plots/detect_outliers/box plot for {col}.png", dpi=300, bbox_inches='tight')
              plt.show()
          else:
              print(f"Column '{col}' has insufficient unique values for a box plot.")


    def create_insightful_visualizations(self):
        """Creates insightful visualizations."""
        if {"TotalPremium", "TotalClaims", "VehicleType"}.issubset(self.data.columns):
            # Visualization 1: Premium vs. Claims by Vehicle Type
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=self.data, x="VehicleType", y="TotalPremium")
            plt.title("Total Premium Distribution by Vehicle Type")
            plt.savefig(f"plots/Insightful/Total Premium by Vehicle type.png", dpi=300, bbox_inches='tight')
            plt.show()

            # Visualization 2: Claims Distribution by CoverType
            plt.figure(figsize=(10, 6))
            sns.barplot(data=self.data, x="CoverType", y="TotalClaims", estimator=sum, ci=None)
            plt.title("Total Claims by CoverType")
            plt.savefig(f"plots/Insightful/Total Claims by CoverType.png", dpi=300, bbox_inches='tight')
            plt.show()

            # Visualization 3: Premiums by Province
            plt.figure(figsize=(10, 6))
            sns.barplot(data=self.data, x="Province", y="TotalPremium", estimator=sum, ci=None)
            plt.title("Total Premiums by Province")
            plt.xticks(rotation=45)
            plt.savefig(f"plots/Insightful/Total Premiums by Province.png", dpi=300, bbox_inches='tight')
            plt.show()
        else:
            print("Columns for insightful visualizations are not available in the dataset.")

