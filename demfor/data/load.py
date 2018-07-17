import pandas as pd
import numpy as np
from demfor.features.date import *


class Dataset:
    def __init__(self, path_to_data="data/raw/", feature_list=None):
        self.path_to_data = path_to_data
        self._load_data()

        self.trainset["id"] = np.arange(self.trainset.shape[0]) + self.testset.shape[0]
        self.train_target = pd.concat([self.trainset["id"], self.trainset["sales"]], axis=1)
        self.trainset = self.trainset[["id", "date", "store", "item"]]

        self.fullset = pd.concat([self.trainset, self.testset], axis=0)

        self.initial_date = datetime.datetime.strptime("2013-01-01", "%Y-%m-%d")  # TODO dynamic

        self.feature_names = list()

        self._create_features()

        self.trainset = self.fullset[self.fullset["id"].isin(self.trainset["id"])]
        self.testset = self.fullset[~self.fullset["id"].isin(self.testset["id"])]

        # self._normalize()

    def _load_data(self):
        self.trainset = pd.read_csv(self.path_to_data + "train.csv")
        self.testset = pd.read_csv(self.path_to_data + "test.csv")

    def reset_data(self):
        self._load_data()

    def _create_features(self):
        # TODO Normalize by cols
        # Mean per store
        # Store cluster?
        # Mean per item?
        # Cluster items?
        self.fullset["date"] = pd.to_datetime(self.fullset["date"], format='%Y-%m-%d')

        # temp, feature_names = day_of_year(self.fullset["date"], self.initial_date)
        # self.fullset = pd.concat([self.fullset, temp], axis=1)
        # self.feature_names += feature_names
        # print("Processed dayofyear")

        # temp, feature_names = day_of_month(self.fullset["date"], self.initial_date)
        # self.fullset = pd.concat([self.fullset, temp], axis=1)
        # self.feature_names += feature_names
        # print("Processed day_of_month")

        temp, feature_names = year(self.fullset["date"], self.initial_date)
        self.fullset = pd.concat([self.fullset, temp], axis=1)
        self.feature_names += feature_names
        print("Processed year")

        # temp, feature_names = month(self.fullset["date"], self.initial_date)
        # self.fullset = pd.concat([self.fullset, temp], axis=1)
        # self.feature_names += feature_names
        # print("Processed month")
        #
        # temp, feature_names = weekday(self.fullset["date"])
        # self.fullset = pd.concat([self.fullset, temp], axis=1)
        # self.feature_names += feature_names
        # print("Processed weekday")
        #
        # temp, feature_names = week_of_year(self.fullset["date"])
        # self.fullset = pd.concat([self.fullset, temp], axis=1)
        # self.feature_names += feature_names
        # print("Processed week_of_year")

    def get_split_in_year_time_series(self):
        X_ts_train = list()
        Y_ts_train = list()
        X_ts_val = list()
        Y_ts_val = list()
        X_ts_test = list()

        for store_id in range(1, 11):
            for item_id in range(1, 51):
                for year in range(2013, 2016):
                    X_index_train = self.trainset[(self.trainset["store"] == store_id) &
                                            (self.trainset["item"] == item_id) &
                                            (self.trainset["year"] == year)]["id"]
                    Y_index_train = self.trainset[(self.trainset["store"] == store_id) &
                                            (self.trainset["item"] == item_id) &
                                            (self.trainset["year"] == year)]["id"]
                    if len(X_index_train) > 365:
                        X_index_train = np.concatenate((X_index_train[:90]), axis=0)
                        # Drop march 31 if needed

                    X_ts_train.append(self.train_target[self.train_target["id"].isin(X_index_train)]["sales"].tolist())
                    Y_ts_train.append(self.train_target[self.train_target["id"].isin(Y_index_train)]["sales"].tolist()[:90])

                X_index_val = self.trainset[(self.trainset["store"] == store_id) &
                                              (self.trainset["item"] == item_id) &
                                              (self.trainset["year"] == 2016)]["id"]
                Y_index_val = self.trainset[(self.trainset["store"] == store_id) &
                                              (self.trainset["item"] == item_id) &
                                              (self.trainset["year"] == 2016)]["id"]
                if len(X_index_val) > 365:
                    X_index_val = np.concatenate((X_index_val[:60], X_index_val[61:]),
                                                 axis=0)  # Pop 29 Feb if needed

                X_ts_val.append(self.train_target[self.train_target["id"].isin(X_index_val)]["sales"].tolist())
                Y_ts_val.append(self.train_target[self.train_target["id"].isin(Y_index_val)]["sales"].tolist()[:90])

                X_index_test = self.trainset[(self.trainset["store"] == store_id) &
                                             (self.trainset["item"] == item_id) &
                                             (self.trainset["year"] == 2017)]["id"]
                if len(X_index_test) > 365:
                    X_index_test = np.concatenate((X_index_test[:60], X_index_test[61:]),
                                                  axis=0)  # Pop 29 Feb if needed

                X_ts_test.append(self.train_target[self.train_target["id"].isin(X_index_test)]["sales"].tolist())




        # TODO one function call
        X_train = np.array(X_ts_train)
        X_train = X_train.reshape((len(X_ts_train), 365, 1))
        Y_train = np.array(Y_ts_train)
        Y_train = Y_train.reshape((len(Y_ts_train), 90))

        X_val = np.array(X_ts_val)
        X_val = X_val.reshape((len(X_ts_val), 365, 1))
        Y_val = np.array(Y_ts_val)
        Y_val = Y_val.reshape((len(Y_ts_val), 90))

        X_test = np.array(X_ts_test)
        X_test = X_test.reshape((len(X_ts_test), 365, 1))

        return X_train, Y_train, X_val, Y_val, X_test

    def _normalize(self):
        # TODO year is not properly normalized
        self.fullset[self.feature_names] = (self.fullset[self.feature_names]) / \
                                           (self.fullset[self.feature_names].max() -
                                            self.fullset[self.feature_names].min())


if __name__ == '__main__':
    dataset = Dataset()

    dataset.fullset.head()