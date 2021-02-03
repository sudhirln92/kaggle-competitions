#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 09:13:14 2021

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

# =============================================================================
# Create Folds
# =============================================================================
class CreateFolds:
    @staticmethod
    def create_folds(kfold=5, source_file="train", save_file="train_folds", seed=42):
        """
        source_file : pass source file name or pandas data frame
        """
        if isinstance(source_file, pd.DataFrame):
            df = source_file
        else:
            df = pd.read_csv(f"input/{source_file}.csv")
        df["kfold"] = -1

        df = df.sample(frac=1).reset_index(drop=True)

        # kfold
        kf = model_selection.KFold(n_splits=kfold, shuffle=True, random_state=seed)
        for fold, (train_idx, val_idx) in enumerate(kf.split(X=df)):
            print(len(train_idx), len(val_idx))
            df.loc[val_idx, "kfold"] = fold

        print("Final shape of file:", df.shape)
        df.to_csv(f"input/{save_file}.csv", index=False)

    @staticmethod
    def read_train_test():
        train = pd.read_csv("input/train.csv")
        test = pd.read_csv("input/test.csv")
        return train, test

    @staticmethod
    def make_xy(train, col, drop_col):
        X = train.drop(drop_col, axis=1)
        y = train[col]
        return X, y

    @staticmethod
    def get_feature_df(X, y, Id, Id_values=None):
        # fit trasform feature pipeline
        pipe = CustomPipeline(pickle_pipe="pipeline")
        X, y = pipe.fit_transform_pipe(X, y)

        # combine data
        df_feat = pd.concat([X, y], axis=1)
        if Id_values is not None:
            df_feat[Id] = Id_values
        return df_feat

    @staticmethod
    def tranform_test_data(
        test,
        save_file,
        target_col,
        Id,
        pipeline_X="pipeline_X",
        pipeline_y="pipeline_y",
    ):
        print("transform test data")
        Id_test = test[Id]
        pipe_file_X = f"models/{pipeline_X}.pkl"
        pipe_X = joblib.load(pipe_file_X)
        X = pipe_X.transform(test)
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if target_col is not None:
            y = test[target_col]
        if pipeline_y is not None:
            pipe_file_y = f"models/{pipeline_y}.pkl"
            pipe_y = joblib.load(pipe_file_y)
            y = pipe_y.transform(y)
            y = pd.DataFrame({target_col: y})

        # save file
        X = pd.DataFrame(X)
        if target_col is not None:
            df = pd.concat([X, y], axis=1)
        else:
            df = X
        df[Id] = Id_test
        df.to_csv(f"input/{save_file}.csv", index=False)
        print("Number of rows and columns in test data", df.shape)

    @classmethod
    def apply_methods(cls):
        # read dataset
        train, test = cls.read_train_test()

        # prepare data
        drop_col = ["target", "id"]
        target_col = ["target"]
        seed = 42
        kfold = 5
        Id = "id"

        X, y = cls.make_xy(train, col=target_col, drop_col=drop_col)

        # fit trasform feature pipeline
        df_feat = cls.get_feature_df(X, y, Id=Id, Id_values=train[Id])

        # create_folds
        cls.create_folds(kfold=kfold, source_file=df_feat, save_file="train_folds")

        # transform test data
        cls.tranform_test_data(
            test,
            target_col=None,
            Id=Id,
            save_file="test_folds",
            pipeline_X="pipeline_X",
            pipeline_y=None,
        )


if __name__ == "__main__":
    # create folds, apply transform on test data
    CreateFolds.apply_methods()
