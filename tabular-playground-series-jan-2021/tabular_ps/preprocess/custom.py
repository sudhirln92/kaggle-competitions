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
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import StandardScaler

# =============================================================================
# Drop Column
# =============================================================================


class DropColumn(TransformerMixin, BaseEstimator):
    """Drop Columns from original Data Frame:

    Parameters
    ----------
    columns: list, The list of column names.

    Returns
    -------
    DataFrame

    Size: (row, old columns - columns)
    """

    def __init__(self, cols):
        if isinstance(cols, str):
            self.cols = [cols]
        else:
            self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        Xdrop = X.drop(self.cols, axis=1)
        return Xdrop


# =============================================================================
# Extract Column
# =============================================================================


class ExtractColumn(TransformerMixin, BaseEstimator):
    """Extract Columns from original Data Frame:

    Parameters
    ----------
    columns: list, The list of column names.

    Returns
    -------
    DataFrame with only extracted columns

    Size: (row, columns)
    """

    def __init__(self, columns, **kwargs):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        Xcol = X[self.columns]
        return Xcol


# =============================================================================
# Custom Standard Scaler
# =============================================================================


class CustomStandardScaler(TransformerMixin, BaseEstimator):
    """Custom Standard scaler which return Pandas DataFrame with column names
    Custom standard scalar will return dataframe with columns, where as
    sklearn standard scalar will return numpy array. The columns name are
    important inorder to analysis data further analysis.
    Parameters
    ----------
    cols: str,list, The list of column names. If column name is none then

    Returns
    -------
    DataFrame with only extracted columns

    Size: (row, columns)
    """

    def __init__(self, cols=None):
        if cols is not None:
            if isinstance(cols, str):
                self.cols = [cols]
            else:
                self.cols = cols
        else:
            self.cols = cols

    def fit(self, X, y=None):
        if self.cols is not None:
            Xo = X[self.cols]
        else:
            Xo = X
            self.cols = X.columns
        self.sc = StandardScaler().fit(Xo)
        return self

    def transform(self, X):
        # filter
        Xo = X[self.cols]

        XScaled = self.sc.transform(Xo)
        XScaled = pd.DataFrame(XScaled, columns=self.cols, index=X.index)
        return XScaled


# =============================================================================
# Custom Feature Union
# =============================================================================


class CustomFeatureUnion(TransformerMixin, BaseEstimator):
    """Custom Feature Union will returns pandas DataFrames
    Custom Feature Union will join two or more dataframe along
    the return dataframe with columns, where as sklearn FeatureUnion
    will return numpy array. The columns name are important inorder
    to analysis data further analysis.

    Parameters
    ----------
    No inputs

    Returns
    -------
    Merge Two or more DataFrames

    Size: (row, columns)
    """

    def __init__(self, transformer_list):
        self.transformer_list = transformer_list

    def fit(self, X, y=None):
        for (name, t) in self.transformer_list:
            t.fit(X, y)
        return self

    def transform(self, X):
        df = pd.DataFrame()
        for (name, t) in self.transformer_list:
            Xts = t.transform(X)
            df = pd.concat([df, Xts], axis=1)
        return df