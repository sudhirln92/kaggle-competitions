#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 09:13:14 2020

@author: sudhir
"""

# =============================================================================
# Import libary
# =============================================================================
import os
import gc
import pandas as pd
import numpy as np
from sklearn import metrics

refresh_log = False
# =============================================================================
# Metric
# =============================================================================


def regression_eval(model, X_train, X_valid, y_train, y_valid):
    # predict
    y_pred_train = model.predict(X_train)
    y_pred = model.predict(X_valid)

    # accuracy score
    trscore = metrics.r2_score(y_train, y_pred_train)
    trscore = round(trscore, 2) * 100
    tscore = metrics.r2_score(y_valid, y_pred)
    tscore = round(tscore, 2) * 100

    mse = metrics.mean_squared_error(y_valid, y_pred)
    rmse = mse ** 0.5
    r2 = metrics.r2_score(y_valid, y_pred)

    # save log
    store_result = f"""
    '{model}'
    Train dataset R square: {trscore}%
    Test dataset R square: {tscore}%
    
    Mean Squared Error(mse): {mse}
    Root Mean Squared Error(rmse): {rmse}
    """
    print(store_result)

    file_name = "model_preds/result.txt"
    if os.path.isfile(file_name) and not refresh_log:
        with open(file_name, "a") as f:
            f.write(store_result)
    else:
        with open(file_name, "w") as f:
            f.write(store_result)

    gc.collect()


def regression_result(y, y_pred, X=None, model=None, printf=False):
    # predict
    if model is not None and X is not None:
        y_pred = model.predict(X)

    # rsquare score
    tscore = metrics.r2_score(y, y_pred)
    tscore = round(tscore, 2) * 100

    mse = metrics.mean_squared_error(y, y_pred)
    rmse = mse ** 0.5

    # save log
    if printf:
        store_result = f"""
        '{model}'
        Test dataset R square\t: {tscore}%
        
        Mean Squared Error(mse) \t: {round(mse,3)}
        Root Mean Squared Error(rmse) \t: {round(rmse,3)}
        """
        print(store_result)

        file_name = "model_preds/prediction.txt"
        if os.path.isfile(file_name) and not refresh_log:
            with open(file_name, "a") as f:
                f.write(store_result)
        else:
            with open(file_name, "w") as f:
                f.write(store_result)
    else:
        return tscore, mse, rmse
