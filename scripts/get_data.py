import numpy as np
import pandas as pd


def get_data():
  df = pd.read_csv("../data/MachineLearningRating_v3.csv", low_memory=False)

  return df

# input_file = "../data/MachineLearningRating_v3.txt"
# output_file = "MachineLearningRating_v3.csv"
# def get_data():
#   df = pd.read_csv(input_file, delimiter="|")
#   df = df.to_csv(f'../data/{output_file}', index=False)
#   # return df




