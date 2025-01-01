import numpy as np
import pandas as pd

df = pd.read_csv('../data/final_cleaned_data.csv')
output_file = '../data/Encoded_data_final.csv'


from scripts.statistical_modeling import StatisticalModeling
stat_model = StatisticalModeling(df, output_file)

# Create pipeline processing for model building and evaluating
def model_builder_evaluator():
  print(f"The dataset has shape of {stat_model.data.shape}")
  stat_model.feature_engineering()

  # Define the features you want to encode
  features_to_encode = ["VehicleType", "make", "Model", "CoverCategory", "CoverType"]


  # Encode categorical columns based on the features list

  stat_model.encode_categorical_data(method='one-hot', features_to_encode=features_to_encode)
  # Split training and testing dat
  stat_model.split_data(target="TotalClaims", test_size=0.2)

  # regression Model
  print("\n=======================================================================================\n")
  print("Regression model using postalCode feature and TotalClaims as target variabla here")
  stat_model.fit_postalcode_models()

  # Step 2: After encoding, update the features list to reflect the one-hot encoded columns
  encoded_features = stat_model.data.columns.tolist()  # Get the list of all column names after encoding
  encoded_features.append('PostalCode')

  # Step 3: Pass the updated features list to build_model
  print("\n=======================================================================================\n")
  print("Optimum regression model training starting here")
  # stat_model.build_model(encoded_features, target="TotalPremium", model_type="random_forest")
  regression_model_types = ['linear_regression', 'decision_tree', 'random_forest']
  task_type = "regression"

  model_types = ['linear_regression', 'decision_tree', 'random_forest']
  stat_model.build_models(features=encoded_features, target="TotalPremium", model_types=regression_model_types, task_type = "regression")
  print("\n=======================================================================================\n")
  print("Optimum classification model training starting here")
  classification_model_types = ['decision_tree', 'random_forest']
  task_type = "classification"
  stat_model.build_models(features=encoded_features, target="TotalPremium", model_types=classification_model_types, task_type = "classification")

  print("\n=======================================================================================\n")
  print("Evaluating models tree")
  model_types = ['linear_regression', 'decision_tree', 'random_forest', 'decision_tree_classification', 'random_forest_classification']

  for model_type in model_types:
    try:
        print(f"Evaluating {model_type}")
        stat_model.evaluate_model(model_type)
    except ValueError as e:
        print(e)

  print("*************************************************************************************************\n")
  print("Feature importance here\n")
  stat_model.analyze_feature_importance(model_type='decision_tree')
  print("\n********************************************************************************************\n")
  print("Interpret with lime package\n")
  stat_model.interpret_with_lime()
  print("\n********************************************************************************************\n")
  print("Compare performance models\n")
  stat_model.compare_model_performance()
