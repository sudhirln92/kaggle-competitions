#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 06 09:13:14 2021


@author: sudhir
"""
# =============================================================================
# Import library
# =============================================================================
import re
import pandas as pd
import numpy as np
from sklearn import model_selection
import joblib

from .pipe import CustomPipeline
from .utils.file_handler import read_config

config = read_config("config.json")

# =============================================================================
# Create Folds
# =============================================================================
class CreateFolds:
    """
    Create Folds and apply feature engineer technique on data

    """

    def __init__(
        self,
        idx,
        kfold,
        seed,
        target_col,
        drop_cols,
        kfold_type="stratified",
        TRAIN_DATA=None,
        TEST_DATA=None,
        **kwargs,
    ):
        # super().__init__()
        self.idx = idx
        self.kfold = kfold
        self.seed = seed
        self.target_col = target_col
        self.drop_cols = drop_cols
        self.TRAIN_DATA = TRAIN_DATA
        self.TEST_DATA = TEST_DATA
        self.kfold_type = kfold_type

    def create_folds(self, source_file, save_file="train_folds"):
        """
        source_file : pandas data frame or read from local folder
        """
        if isinstance(source_file, pd.DataFrame):
            df = source_file
        else:
            df = pd.read_csv(f"input/{self.TRAIN_DATA}.csv")
        df["kfold"] = -1

        df = df.sample(frac=1).reset_index(drop=True)

        # kfold
        if self.kfold_type == "stratified":
            y = df[self.target_col]
            y = pd.cut(y, self.kfold, labels=False)
            kf = model_selection.StratifiedKFold(
                n_splits=self.kfold, random_state=self.seed, shuffle=True
            )
            for fold, (train_idx, val_idx) in enumerate(kf.split(X=df, y=y)):
                print("stratified", len(train_idx), len(val_idx))
                df.loc[val_idx, "kfold"] = fold
        else:

            kf = model_selection.KFold(n_splits=self.kfold, random_state=self.seed)
            for fold, (train_idx, val_idx) in enumerate(kf.split(X=df)):
                print(len(train_idx), len(val_idx))
                df.loc[val_idx, "kfold"] = fold

        print("Final shape of file:", df.shape)
        df.to_csv(f"input/{save_file}.csv", index=False)

    def get_feature_df(self, train):
        # split data
        X = train.drop(self.drop_cols, axis=1)
        y = train[self.target_col]
        Id_values = train[self.idx]

        # fit trasform feature pipeline
        pipe = CustomPipeline(pickle_pipe="pipeline")
        X, y = pipe.fit_transform_pipe(X, y)

        if isinstance(y, pd.Series) or isinstance(y, np.ndarray):
            y = pd.DataFrame(y, columns=[self.target_col])

        # combine data
        df_feat = pd.concat([X, y], axis=1)
        if Id_values is not None:
            df_feat[self.idx] = Id_values

        return df_feat

    def tranform_test_data(self, test, save_file, pipeline):
        print("transform test data...")
        Id_test = test[self.idx]
        pipe_file_X = f"models/{pipeline}_X.pkl"
        pipe_X = joblib.load(pipe_file_X)
        X = pipe_X.transform(test)
        X[self.idx] = Id_test

        # save
        X.to_csv(f"input/{save_file}.csv", index=False)
        print("Number of rows and columns in test data", X.shape)

    def read_train_test(self, nrows):
        """nrows : None read all rows"""
        train = pd.read_csv(f"input/{self.TRAIN_DATA}", nrows=nrows)
        test = pd.read_csv(f"input/{self.TEST_DATA}", nrows=nrows)
        return train, test

    def combine_train_test(self, train, test):
        """combine train test dataset"""
        train["data"] = "train"

        test[self.target_col] = np.nan
        test["data"] = "test"

        df = pd.concat([train, test], axis=0).reset_index(drop=True)

        return df

    def split_train_test(self, df):

        train = df.query("data == 'train' ").drop("data", axis=1).reset_index(drop=True)

        test = (
            df.query("data == 'test' ")
            .drop(["data", self.target_col], axis=1)
            .reset_index(drop=True)
        )

        return train, test

    def apply_methods(self):
        nrows = None  # None / 2000
        # read dataset
        train, test = self.read_train_test(nrows)

        # combine_train_test
        df = self.combine_train_test(train, test)

        #  fit trasform feature pipeline
        df_feat = self.get_feature_df(df)

        # split_train_test
        train_feat, test_feat = self.split_train_test(df_feat)

        # create_folds
        self.create_folds(source_file=train_feat, save_file="train_folds")

        # transform test data
        # self.tranform_test_data(test, save_file="test_folds", pipeline="pipeline")
        test_feat.to_csv(f"input/test_folds.csv", index=False)
        print("Number of rows and columns in test data", test_feat.shape)


if __name__ == "__main__":
    # create folds, apply transform on test data

    CreateFolds(**config).apply_methods()
