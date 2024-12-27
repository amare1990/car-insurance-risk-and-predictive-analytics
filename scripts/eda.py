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
