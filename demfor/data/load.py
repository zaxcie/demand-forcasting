import pandas as pd
import numpy as np
from demfor.features.date import *


class Dataset:
    def __init__(self, path_to_data="data/raw/", feature_list=None):

        self.trainset = pd.read_csv(path_to_data + "train.csv")
        self.testset = pd.read_csv(path_to_data + "test.csv")

        self.initial_date = datetime.datetime.strptime("2013-01-01", "%Y-%m-%d")  # TODO dynamic

        # Exec features
        if feature_list is None:
            self.testset["date"] = pd.to_datetime(self.testset["date"], format='%Y-%m-%d')
            self.testset["dayofyear"], self.testset["dayofyear_elapse"] = day_of_year(self.testset["date"],
                                                                                      self.initial_date)
            self.testset = pd.concat([self.testset, day_of_month(self.testset["date"], self.initial_date)], axis=1)
            self.testset = pd.concat([self.testset, year(self.testset["date"], self.initial_date)], axis=1)
            self.testset = pd.concat([self.testset, month(self.testset["date"], self.initial_date)], axis=1)
            self.testset = pd.concat([self.testset, weekday(self.testset["date"])], axis=1)
            self.testset = pd.concat([self.testset, week_of_year(self.testset["date"])], axis=1)

            # TODO Normalize by cols
            # Mean per store
            # Store cluster?
            # Mean per item?
            # Cluster items?



if __name__ == '__main__':
    dataset = Dataset()

    dataset.trainset.head()