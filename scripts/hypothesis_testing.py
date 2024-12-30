
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, chi2_contingency

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

    def segment_data(self, feature: str, group_a_value, group_b_value):
        """
        Segment data into control (Group A) and test (Group B) groups based on a feature.
        :param feature: The column name to segment by.
        :param group_a_value: The value to select Group A.
        :param group_b_value: The value to select Group B.
        """
        if feature not in self.data.columns:
            raise ValueError(f"{feature} is not a valid column in the dataset.")

        self.group_a = self.data[self.data[feature] == group_a_value]
        self.group_b = self.data[self.data[feature] == group_b_value]

        if self.group_a.empty or self.group_b.empty:
            raise ValueError("One of the groups is empty. Ensure valid segmentation.")

        print(f"Data segmented by {feature}: Group A ({group_a_value}), Group B ({group_b_value})")


    def perform_statistical_test(self, test_type: str = "t-test"):
        """
        Perform statistical tests on the segmented data.
        :param test_type: Type of test to perform ("t-test" or "chi-squared").
        :return: p-value of the test.
        """
        if not hasattr(self, 'group_a') or not hasattr(self, 'group_b'):
            raise AttributeError("Data segmentation must be performed before testing.")

        if test_type == "t-test":
            stat, p_value = ttest_ind(self.group_a[self.kpi], self.group_b[self.kpi], nan_policy='omit')
        elif test_type == "chi-squared":
            contingency_table = pd.crosstab(self.group_a[self.kpi], self.group_b[self.kpi])
            stat, p_value, _, _ = chi2_contingency(contingency_table)
        else:
            raise ValueError(f"Unsupported test type: {test_type}")

        print(f"Performed {test_type}: p-value = {p_value}")
        return p_value

    def analyze_results(self, p_value, alpha=0.05):
        """
        Analyze statistical test results and interpret the outcome.
        :param p_value: P-value from the statistical test.
        :param alpha: Significance level (default = 0.05).
        """
        if p_value < alpha:
            print("Reject the null hypothesis: The feature has a statistically significant effect.")
        else:
            print("Fail to reject the null hypothesis: The feature does not have a significant effect.")
