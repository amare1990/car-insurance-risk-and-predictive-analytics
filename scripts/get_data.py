import numpy as np
import pandas as pd


def get_data():
  df = pd.read_csv("../data/MachineLearningRating_v3.txt", delimiter="|", header=0)

  return df


df_from_text  = get_data()

df = pd.to_csv('../data/df_from_text')
print(df.head())
