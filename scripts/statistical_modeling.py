import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Import feature importance and model interpretability analysis modules
import shap
import lime
from lime.lime_tabular import LimeTabularExplainer


# from scripts.data_cleaning import DataCleaning

class StatisticalModeling:
    def __init__(self, data: pd.DataFrame, output_file="../data/Encoded_data.csv"):
        """
        Initialize the class with the dataset.
        :param data: A pandas DataFrame containing the dataset.
        """
        # input_file="../data/MachineLearningRating_v3.csv"
        # output_file="../data/final_cleaned_data_stmodeling.csv"
        # data_cleaner = DataCleaning(input_file, output_file)
        # data = data_cleaner.process_pipeline()

        self.data = data
        self.output_file = output_file
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}


    def feature_engineering(self):
        """
        Create new features relevant to the prediction of TotalPremium and TotalClaims.
        """
        # Example of a new feature: creating a feature for premium per claim
        self.data['PremiumPerClaim'] = self.data['TotalPremium'] / (self.data['TotalClaims'] + 1)
        print("Created new feature: PremiumPerClaim")

    def encode_categorical_data(self, method='one-hot', features_to_encode=None):
        """
        Encodes categorical features in the dataset.
        :param method: Encoding method ('one-hot' or 'label').
        :param features_to_encode: List of columns to encode. If None, all categorical columns will be encoded.
        """
        # Identify date-like columns and exclude them
        date_columns = []
        for col in self.data.columns:
            try:
                pd.to_datetime(self.data[col], errors='coerce')  # Check if column is date-like
                date_columns.append(col)
            except Exception:
                continue

        # Determine features to encode if not explicitly provided
        if features_to_encode is None:
            features_to_encode = self.data.select_dtypes(include=['object']).columns.tolist()

        # Exclude date-like columns from encoding
        features_to_encode = [col for col in features_to_encode if col not in date_columns]

        # Print info about columns being encoded
        print(f"Columns to encode: {features_to_encode}")
        print(f"Date-like columns excluded: {date_columns}")

        # Perform encoding
        if method == 'one-hot':
            self.data = pd.get_dummies(self.data, columns=features_to_encode, drop_first=True)
        elif method == 'label':
            label_enc = LabelEncoder()
            for col in features_to_encode:
                self.data[col] = label_enc.fit_transform(self.data[col])
        else:
            raise ValueError(f"Unsupported encoding method: {method}")



        # Saving the encoded data
        output_file = "../data/Encoded_data.csv"
        self.output_file = output_file
        self.data.to_csv(self.output_file, index=False)
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
        print(f"Data split into {len(self.X_train)} train samples and {len(self.X_test)} test samples.")

    def fit_postalcode_models(self, postalcode_column="PostalCode", target="TotalClaims"):
        postalcode_models = {}
        unique_postalcodes = self.data[postalcode_column].unique()

        for postalcode in unique_postalcodes:
            postalcode_data = self.data[self.data[postalcode_column] == postalcode]
            if len(postalcode_data) < 10:  # Ensure enough data points for modeling
                continue

            X = postalcode_data.drop(columns=[target, postalcode_column])
            y = postalcode_data[target]

            # Ensure X is numeric
            X = X.select_dtypes(include=['float64', 'int64'])

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = LinearRegression()
            model.fit(X_train, y_train)
            postalcode_models[postalcode] = model

        print(f"Fitted linear regression models for {len(postalcode_models)} postalcodes.")
        return postalcode_models

    def build_models(self, features, target, model_types=None, task_type="regression"):
        """
        Build and train models based on the task type (regression or classification).
        :param features: DataFrame containing features.
        :param target: Target column name.
        :param model_types: List of model types to train.
        :param task_type: Type of task ('regression' or 'classification').
        :return: Dictionary of trained models.
        """
        models = {}
        X = self.data.drop(columns=[target])
        y = self.data[target]

        # For classification task, create target variable for profit/loss
        if task_type == "classification":
            features['profit_loss'] = (features['TotalPremium'] - features['TotalClaims']).apply(
                lambda x: 'profit' if x > 0 else 'loss')
            y = features['profit_loss']  # Update target variable to profit_loss

            unique_classes = len(y.unique())
            print(f"Detected classification task with {unique_classes} classes.")
            y = y.astype('category') if unique_classes > 2 else y
        else:
            print("Detected regression task.")

        # Step 4: Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        print(f"Data split into training and testing sets: {len(X_train)} training samples, {len(X_test)} testing samples.")

        for model_type in model_types:
            print(f"Training {model_type} model...")

            # Select model based on task type
            if task_type == "regression":
                print(f"Regression model, {model_type} being trained")
                if model_type == 'linear_regression':
                    model = LinearRegression()
                elif model_type == 'decision_tree':
                    model = DecisionTreeRegressor()
                elif model_type == 'random_forest':
                    model = RandomForestRegressor()
                elif model_type == 'xgboost':
                    model = xgb.XGBRegressor()
                else:
                    print(f"Unsupported regression model: {model_type}")
                    continue
            elif task_type == "classification":
                print(f"Classification model, {model_type} being trained")
                if model_type == 'decision_tree':
                    model = DecisionTreeClassifier()
                elif model_type == 'random_forest':
                    model = RandomForestClassifier()
                elif model_type == 'xgboost':
                    model = xgb.XGBClassifier()
                else:
                    print(f"Unsupported classification model: {model_type}")
                    continue
            else:
                print(f"Unsupported task type: {task_type}")
                continue

            # Train the model
            model.fit(self.X_train, y_train)
            models[model_type] = model
            print(f"{model_type} model trained successfully.")

        return models



    def evaluate_model(self, model_type):
        """
        Evaluate the performance of a trained model.
        :param model_type: The model type to evaluate.
        """
        # Retrieve the model
        model = self.models.get(model_type)
        if model is None:
            raise ValueError(f'Model {model_type} has not been trained yet!')

        # Make predictions
        y_pred = model.predict(self.X_test)

        if isinstance(model, (LinearRegression, DecisionTreeRegressor, RandomForestRegressor, xgb.XGBRegressor)):
            # Regression evaluation
            print(f"Regression evaluation for {model_type} taking place...")
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            print(f'{model_type} Evaluation:')
            print(f'MSE: {mse}')
            print(f'RMSE: {rmse}')
            return {"MSE": mse, "RMSE": rmse}

        elif isinstance(model, (DecisionTreeClassifier, RandomForestClassifier, xgb.XGBClassifier)):
            # Classification evaluation
            print(f"Classification evaluation for {model_type} taking place...")

            if len(self.y_test.unique()) > 2:  # Multiclass
                predictions = np.argmax(y_pred, axis=1) if hasattr(model, "predict_proba") else y_pred
                average = 'weighted'
            else:  # Binary classification
                predictions = (y_pred > 0.5).astype(int) if hasattr(model, "predict_proba") else y_pred
                average = 'binary'

            accuracy = accuracy_score(self.y_test, predictions)
            precision = precision_score(self.y_test, predictions, average=average, zero_division=1)
            recall = recall_score(self.y_test, predictions, average=average, zero_division=1)
            f1 = f1_score(self.y_test, predictions, average=average, zero_division=1)

            print(f'Model {model_type} - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1_score: {f1}')
            return {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1-score": f1}

        else:
            raise ValueError(f"Unsupported model type: {model_type}")



    def analyze_feature_importance(self, model_type='random_forest'):
        """
        Analyze feature importance using SHAP for the specified model type.
        :param model_type: The model type to analyze.
        """
        model = self.models.get(model_type)
        if model is None:
            raise ValueError(f'Model {model_type} has not been trained yet!')

        # Use SHAP to explain predictions
        explainer = shap.TreeExplainer(model) if model_type in ['random_forest', 'decision_tree'] else shap.Explainer(model)
        shap_values = explainer.shap_values(self.X_test)

        # Summary plot
        print(f"Generating SHAP summary plot for {model_type}...")
        shap.summary_plot(shap_values, self.X_test, plot_type="bar")


    def interpret_with_lime(self, model_type='random_forest', sample_index=0):
        """
        Interpret model predictions using LIME for a specific instance.
        :param model_type: The model type to interpret.
        :param sample_index: Index of the sample to interpret from the test set.
        """
        model = self.models.get(model_type)
        if model is None:
            raise ValueError(f'Model {model_type} has not been trained yet!')

        # Prepare the LIME explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=self.X_train.values,
            feature_names=self.X_train.columns.tolist(),
            class_names=['Target'],
            mode='regression' if model_type in ['linear_regression'] else 'classification'
        )

        sample = self.X_test.iloc[sample_index].values
        explanation = explainer.explain_instance(sample, model.predict, num_features=10)

        # Display explanation
        explanation.show_in_notebook()
        explanation.as_pyplot_figure()
        plt.show()

    def compare_model_performance(self):
        """
        Compare the performance of all trained models.
        """
        if not self.models:
            print("No models have been trained yet!")
            return

        performance_metrics = {}
        for model_type, model in self.models.items():
            print(f"Evaluating performance for {model_type}...")
            metrics = self.evaluate_model(model_type=model_type)
            performance_metrics[model_type] = metrics

        # Report performance comparison
        print("\nModel Performance Comparison:")
        for model_type, metrics in performance_metrics.items():
            print(f"{model_type}: {metrics}")

        return performance_metrics




