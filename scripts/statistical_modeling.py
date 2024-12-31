import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
# import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


from scripts.data_cleaning import DataCleaning

class StatisticalModeling:
    def __init__(self):
        """
        Initialize the class with the dataset.
        :param data: A pandas DataFrame containing the dataset.
        """
        input_file="../data/MachineLearningRating_v3.csv"
        output_file="../data/final_cleaned_data_stmodeling.csv"
        data_cleaner = DataCleaning(input_file, output_file)
        data = data_cleaner.process_pipeline()

        self.data = data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}


    def handle_missing_data(self, threshold=0.3):
        """
        Handle missing values:
        - Drop columns with > threshold missing values.
        - Impute remaining missing values with mean (numerical) or mode (categorical).
        :param threshold: The threshold for the percentage of missing values allowed in columns.
        """
        # Drop columns with > threshold missing values
        missing_percent = self.data.isnull().mean()
        columns_to_drop = missing_percent[missing_percent > threshold].index.tolist()
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

    def feature_engineering(self):
        """
        Create new features relevant to the prediction of TotalPremium and TotalClaims.
        """
        # Example of a new feature: creating a feature for premium per claim
        self.data['PremiumPerClaim'] = self.data['TotalPremium'] / (self.data['TotalClaims'] + 1)
        print("Created new feature: PremiumPerClaim")

    def encode_categorical_data(self, method='one-hot'):
        categorical_columns = self.data.select_dtypes(include=['object']).columns
        if method == 'one-hot':
            self.data = pd.get_dummies(self.data, columns=categorical_columns, drop_first=True)
        elif method == 'label':
            label_enc = LabelEncoder()
            for col in categorical_columns:
                self.data[col] = label_enc.fit_transform(self.data[col])
        else:
            raise ValueError(f"Unsupported encoding method: {method}")
        print(f"Encoded categorical columns using {method} encoding.")

    def split_data(self, target="TotalClaims", test_size=0.2):
        """
        Split the dataset into training and testing sets.
        :param target: The column to predict (usually 'TotalClaims' or 'TotalPremium').
        :param test_size: The proportion of the data to use for testing.
        """
        X = self.data.drop(columns=[target])
        y = self.data[target]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        print(f"Data split into {len(self.X_train)} train samples and
              {len(self.X_test)} test samples.")


