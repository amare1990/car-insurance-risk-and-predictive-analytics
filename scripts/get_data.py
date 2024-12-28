""" A method that encapsulated data and provides pandas dataframe to the EDA class"""
import pandas as pd


def get_data():
    """
    Converts the csv file into a Pandas dataframe
    returns a pandas dataframe
    """
    df = pd.read_csv("../data/MachineLearningRating_v3.csv", low_memory=False)

    return df
