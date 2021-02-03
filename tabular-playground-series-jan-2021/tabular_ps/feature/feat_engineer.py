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
import geopandas as gpd
from shapely import geometry
from sklearn.base import TransformerMixin, BaseEstimator


# =============================================================================
# Dummy variable creator / One hot encoding
# =============================================================================


class DummyTransformer(TransformerMixin, BaseEstimator):
    """Dummy Varaible Transform / One Hot Encoding:

    Parameters
    ----------
    columns: list, input Required columns name to get dummay variables


    Returns
    -------
    DataFrame with column names

    Size: (row, new_columns + old columns)
    """

    def __init__(self, columns, **kwargs):
        self.columns = columns

    def fit(self, X, y=None):
        self.cat = {}
        for col in self.columns:
            self.cat[col] = X[col].astype("category").cat.categories
        return self

    def transform(self, X):

        Xtmp = X[self.columns].copy()

        # transform categorical
        for col in self.columns:
            Xtmp[col] = pd.Categorical(Xtmp[col], categories=self.cat[col])
        Xcat = pd.get_dummies(Xtmp, columns=self.columns, drop_first=True)

        # Merge dataset
        Xtmp = pd.concat([X, Xcat], axis=1)
        return Xtmp