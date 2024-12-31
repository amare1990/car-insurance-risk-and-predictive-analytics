# Insert the parent directory to the system path
import os, sys
curr_dir = os.getcwd()
parent_dir = os.path.dirname(curr_dir)
sys.path.insert(0, parent_dir)

import numpy as np
import pandas as pd

df = pd.read_csv('../data/final_cleaned_data.csv')
output_file="../data/Encoded_data.csv"


from scripts.statistical_modeling import StatisticalModeling
stat_model = StatisticalModeling(df, output_file)

# Create pipeline processing for model building and evaluating
def model_builder_evaluator():
  print(f"The dataset has shape of {stat_model.data.shape}")
  stat_model.feature_engineering()
  # stat_model.encode_categorical_data()
  # stat_model.split_data(target="TotalClaims", test_size=0.2)
  # stat_model.build_model(model_type='linear_regression')
  # stat_model.evaluate_model(model_type='linear_regression')
