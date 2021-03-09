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
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer


# =============================================================================
# Handle Outlier
# =============================================================================


class HandleOutlier(TransformerMixin, BaseEstimator):
    """HandleOutlier in columns:

    Parameters
    ----------
    Input dataframe,
    columns names

    Ex : cols = ['PickUpLat','PickUpLat']
    Returns
    -------
    DataFrame

    Size: (row, columns)
    """

    def __init__(self, cols, **kwargs):
        if isinstance(cols, str):
            self.cols = [cols]
        else:
            self.cols = cols

    def fit(self, X, y=None):
        # compute
        self.lower_bound = {}
        self.upper_bound = {}
        for col in self.cols:
            low, up = self.compute_bound(X, col)
            self.lower_bound[col] = low
            self.upper_bound[col] = up
        return self

    def compute_bound(self, X, col):
        """Compute lower bound and upper bound for upper bound
        for outlier detection
        """
        q1 = X[col].quantile(0.25)
        q2 = X[col].quantile(0.75)
        iqr = q2 - q1
        low = q1 - 1.5 * iqr
        up = q2 + 1.5 * iqr
        return low, up

    def transform(self, X):
        # print('Limit', self.limit)
        for col in self.cols:
            # Lower than limit
            X[col] = np.where(X[col] > self.lower_bound[col], X[col], np.nan)
            # X.loc[(X[col]) < self.lower_bound[col], col] = np.nan
            # Greater than limit
            X[col] = np.where(X[col] < self.upper_bound[col], X[col], np.nan)
            # X.loc[(X[col]) > self.upper_bound[col], col] = np.nan

        return X


# =============================================================================
# CustomQuantileTransformer
# =============================================================================


class CustomQuantileTransformer(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        cols=None,
        n_quantiles=1000,
        output_distribution="normal",
        random_state=42,
        **kwargs,
    ):
        """
        cols: pass column names
        n_quantiles:
        """
        if isinstance(cols, str):
            self.cols = [cols]
        else:
            self.cols = cols
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution
        self.random_state = random_state

    def fit(self, X, y=None):
        """
        fit
        """
        self.quant_trans = QuantileTransformer(
            n_quantiles=self.n_quantiles,
            output_distribution=self.output_distribution,
            random_state=self.random_state,
        )
        if isinstance(X, pd.DataFrame):
            self.quant_trans.fit(X[self.cols])
        elif isinstance(X, np.ndarray):
            self.quant_trans.fit(X)
        else:
            raise ValueError("input should be DataFrame or array")
        return self

    def transform(self, X):
        """
        transform
        """
        if isinstance(X, pd.DataFrame):
            Xo = self.quant_trans.transform(X[self.cols])
            Xo = pd.DataFrame(Xo, columns=self.cols)
            Xo = pd.concat([X.drop(self.cols, axis=1), Xo], axis=1)
        elif isinstance(X, np.ndarray):
            Xo = self.quant_trans.transform(X)
        else:
            raise ValueError("input should be DataFrame or array")
        return Xo

    def inverse_transform(self, X):
        """
        inverse_transform
        """
        if isinstance(X, pd.DataFrame):
            Xo = self.quant_trans.inverse_transform(X[self.cols])
            Xo = pd.DataFrame(Xo, columns=self.cols)
            Xo = pd.concat([X.drop(self.cols, axis=1), Xo], axis=1)
        elif isinstance(X, np.ndarray):
            Xo = self.quant_trans.inverse_transform(X)
        else:
            raise ValueError("input should be DataFrame or array")

        return Xo