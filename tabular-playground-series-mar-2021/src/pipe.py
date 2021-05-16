#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 09:13:14 2021

@author: sudhir
"""
# =============================================================================
# Import library
# =============================================================================
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.mixture import GaussianMixture
import joblib

from .preprocess import preprocess, custom
from .feature import feature, feat_engineer

seed = 42

# =============================================================================
# pipe
# =============================================================================


class CustomPipeline:
    """
    Custom Pipeline
    """

    def __init__(self, pickle_pipe=None):
        """
        pickle_pipe : file name with path
        """
        self.pickle_pipe = pickle_pipe

        self.num_cols = [
            "cont0",
            "cont1",
            "cont2",
            "cont3",
            "cont4",
            "cont5",
            "cont6",
            "cont7",
            "cont8",
            "cont9",
            "cont10",
        ]
        self.cat_cols = [
            "cat0",
            "cat1",
            "cat2",
            "cat3",
            "cat4",
            "cat5",
            "cat6",
            "cat7",
            "cat8",
            "cat9",
            "cat10",
            "cat11",
            "cat12",
            "cat13",
            "cat14",
            "cat15",
            "cat16",
            "cat17",
            "cat18",
        ]

        # type conversion
        self.dtype_param = {
            "cat_cols": self.cat_cols,
            "num_cols": self.num_cols,
        }

        self.outlier_param = {"cols": self.num_cols}

        # impute_param
        self.impute_param = {
            "cat_cols": self.cat_cols,
            "num_cols": self.num_cols,
            "date_cols": [],
            "dont_use_cols": [],
            "n_estimators": 100,
            "max_depth": 10,
        }

        # self.gmm_param
        self.gmm_param = {
            "cols_component": {
                "cont1": 4,
                "cont2": 10,
                "cont3": 6,
                "cont4": 4,
                "cont5": 3,
                "cont6": 2,
                "cont7": 3,
                "cont8": 4,
                "cont9": 4,
                "cont10": 8,
            },
            "col_suffix": "_gmm",
            "predict_proba": False,
        }
        # quantile transform
        self.quant_param = {
            "cols": self.num_cols,
            "n_quantiles": 1000,
            "output_distribution": "normal",
            "random_state": seed,
        }

        self.dummy_param = [
            "cat0",
            "cat1",
            "cat2",
            "cat3",
            "cat4",
            "cat5",
            "cat6",
            "cat7",
            "cat8",
            "cat9",
            "cat10",
            "cat11",
            "cat12",
            "cat13",
            "cat14",
            "cat15",
            "cat16",
            "cat17",
            "cat18",
        ]
        self.drop_columns = self.dummy_param

    def feature_pipe(self):
        """
        # Feature Engineering Pipeline
        # Step: ExtractColumn
        # Step: QuantileTransformer
        # Step: Multi GaussianEncoder
        """

        print("-" * 20, "Feature Engineering Pipeline")
        pipe = Pipeline(
            [
                # ("ExtractColumn", custom.ExtractColumn(self.extract_param)),
                # ("HandleOutlier", preprocess.HandleOutlier(**self.outlier_param)),
                ("gmm_feat", feature.CustomGMM(**self.gmm_param)),
                ("feq combine", feat_engineer.CombineCatFeat(self.dummy_param)),
                ("numeric", feat_engineer.NumericFeat(self.num_cols)),
                # ("DummyVariable", feat_engineer.DummyTransformer(self.dummy_param)),
                ("label encode", feat_engineer.CustomLabelEncoder(self.dummy_param)),
                ("Quan", preprocess.CustomQuantileTransformer(**self.quant_param)),
                # ("Drop Columns1", custom.DropColumn(self.drop_columns)),
            ]
        )

        return pipe

    def fit_transform_pipe(self, X, y=None):
        self.pipe_X = self.feature_pipe()

        # fit transform
        X = self.pipe_X.fit_transform(X, y)
        if isinstance(self.pickle_pipe, str):
            joblib.dump(self.pipe_X, f"models/{self.pickle_pipe}_X.pkl")
        return X, y

    def transform_y(self, y, target_cols):
        for col in target_cols:
            quant = joblib.load(f"models/{self.pickle_pipe}_{col}.pkl")
            y_hat = y[col].values.reshape(1, -1)
            y[col] = quant.transform(y_hat)
        return y

    def transform_pipe(self, X, y=None):
        X = self.pipe_X.transform(X)
        target_cols = ["target"]
        return X, y
