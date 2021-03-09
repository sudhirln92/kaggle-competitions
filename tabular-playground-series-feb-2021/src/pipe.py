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

        # extract_param
        self.extract_param = [
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
            "cont11",
            "cont12",
            "cont13",
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
        ]
        num_cols = [
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
            "cont11",
            "cont12",
            "cont13",
        ]
        cat_cols = [
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
        ]

        # type conversion
        self.dtype_param = {
            "cat_cols": cat_cols,
            "num_cols": num_cols,
        }

        self.outlier_param = {"cols": num_cols}

        # impute_param
        self.impute_param = {
            "cat_cols": cat_cols,
            "num_cols": num_cols,
            "date_cols": [],
            "dont_use_cols": [],
            "n_estimators": 100,
            "max_depth": 10,
        }

        self.kmeans_param = {
            "cluster_prefix": "cluster",
            "seed": seed,
            "n_clusters": 25,
            "columns": ["PickUpLon", "PickUpLat"],
        }

        self.pca_param = {
            "n_components": 4,
            "cols": self.extract_param,
            "col_prefix": "pca",
            "seed": seed,
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
                "cont11": 5,
                "cont12": 4,
                "cont13": 6,
            },
            "col_suffix": "_gmm",
            "predict_proba": True,
        }
        # quantile transform
        self.quant_param = {
            "cols": num_cols,
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
        ]
        self.drop_columns = self.dummy_param

    def feature_pipe(self):
        """
        # Feature Engineering Pipeline
        # Step: ExtractColumn
        # Step: Data Preprocessing
        # Step: HandleOutlier
        # Step: QuantileTransformer
        # Step: Multi GaussianEncoder
        """

        print("-" * 20, "Feature Engineering Pipeline")
        pipe = Pipeline(
            [
                ("ExtractColumn", custom.ExtractColumn(self.extract_param)),
                # ("HandleOutlier", preprocess.HandleOutlier(**self.outlier_param)),
                # ("Impute", custom_impute.MissForestImputer(**self.impute_param)),
                # ("KMeans p", unsupervised.CustomKMeans(**self.kmeans_pickup_param)),
                # ("Gaussian", encoder.MultiGaussianEncoder(**self.gaussian_param)),
                # ("PCA", unsupervised.CustomPCA(**self.pca_param)),
                # ("gmm_feat", feature.CustomGMM(**self.gmm_param)),
                ("DummyVariable", feat_engineer.DummyTransformer(self.dummy_param)),
                ("Quan", preprocess.CustomQuantileTransformer(**self.quant_param)),
                ("Drop Columns1", custom.DropColumn(self.drop_columns)),
            ]
        )

        return pipe

    def fit_y(self, y):
        target_cols = ["target"]
        quant_param = {
            "n_quantiles": 5000,
            "output_distribution": "normal",
            "random_state": seed,
        }

        for col in target_cols:
            quant = preprocess.CustomQuantileTransformer(**quant_param)
            y_hat = y[col].values.reshape(-1, 1)
            quant.fit(y_hat)
            joblib.dump(quant, f"models/{self.pickle_pipe}_{col}.pkl")
            y_res = quant.transform(y_hat)
            y[col] = y_res.reshape(-1, 1)[:, 0]

        return y

    def fit_transform_pipe(self, X, y=None):
        self.pipe_X = self.feature_pipe()

        # y = self.fit_y(y)

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
        # y = self.transform_y(y, target_cols)

        return X, y
