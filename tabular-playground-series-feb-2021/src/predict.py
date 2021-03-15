#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 09:13:14 2021

@author: sudhir
"""
# =============================================================================
# Import library
# =============================================================================
import os
import pandas as pd
import numpy as np
import joblib
from .skmetrics import regression_result
from . import dispatcher

MODELS_FOLDER = "models"
FEATURE_PIPELINE = "pipeline_X.pkl"
TARGET_PIPELINE = "pipeline_target.pkl"
TEST_DATA = "input/test_folds.csv"

# =============================================================================
# Predict
# =============================================================================


class RegressionPredict:
    """
    Regression Predict
    """

    def __init__(self, Id, kfold, drop_cols, feature_pipeline_pkl, **kwargs):
        self.Id = Id
        self.kfold = kfold
        self.drop_cols = drop_cols
        self.feature_pipeline_pkl = feature_pipeline_pkl
        self.models_folder = "models"

    def inverse_transform_y(self, pred, target_col):
        # pipe y
        file = f"{self.models_folder}/pipeline_{target_col}.pkl"
        pipe_y = joblib.load(file)
        pred = pipe_y.inverse_transform(pred.reshape(-1, 1))
        return pred

    def predict_folds_model_avg(
        self, model_name, target_col, X, y=None, print_cv=False
    ):
        # predict proba for the regression
        y_pred = np.zeros((X.shape[0], 1))
        for f in range(self.kfold):
            # read model file
            model_file = f"{self.models_folder}/{model_name}_{f}.pkl"
            clf = joblib.load(model_file)

            # predict
            pred = clf.predict(X)
            y_pred = y_pred + pred.reshape(-1, 1)

            # cross valid prediction
            if print_cv:
                regression_result(y, pred, printf=True)

        # average prediction of models
        y_pred = y_pred / self.kfold
        return y_pred

    def multi_model_prediction(self, df_test, MODELS, target_col, last_stage=False):
        pred = []
        X = df_test.drop(self.drop_cols, axis=1)

        # multi model prediction
        for model_name in MODELS:
            # predict
            y_pred = self.predict_folds_model_avg(model_name, target_col, X)

            pred.append(pd.DataFrame(y_pred, columns=[model_name]))
        pred = pd.concat(pred, axis=1)
        pred = pred.mean(axis=1)
        return pred

    def single_model_prediction(self, df_test, MODELS, last_stage=False):
        # single model
        X = df_test.drop(self.drop_cols, axis=1)

        # multi model prediction
        for model_name in MODELS:
            target_col = "target"
            # predict
            y_pred = self.predict_folds_model_avg(model_name, target_col, X)

            # print result
            # pred_tmp = self.inverse_transform_y(y_pred, target_col)
            # if last_stage:
            #    y_pred = pred_tmp
        pred = pd.DataFrame(y_pred, columns=["target"])
        return pred

    def final_predict(self):
        # read test data
        df_test = pd.read_csv(TEST_DATA)

        # predict_every_model
        pred = self.single_model_prediction(
            df_test, MODELS=dispatcher.MODELS.keys(), last_stage=True
        )
        pred[self.Id] = df_test[self.Id]

        # stage 2 prediction
        # df_test = pd.merge(df_test, pred, on=self.Id, how="left")

        # Id = df_test[self.Id]
        # pred = self.multi_model_prediction(
        #     df_test, MODELS=dispatcher.MODELS_STAGE2.keys(), last_stage=True
        # )
        # pred[self.Id] = Id

        return pred

    def prepare_data(self, df_test):
        test = pd.read_csv("input/test.csv")
        if test.shape[0] == 0:
            raise ValueError("Test data as ZERO rows")
        return test

    def predict_unseen(self, df):
        # load pickle pipeline
        pipe_file_X = f"{self.models_folder}/{self.feature_pipeline_pkl}"
        pipe_X = joblib.load(pipe_file_X)

        # apply trasform
        Id = df[self.Id]
        X = pipe_X.transform(df)

        # predict_every_model
        y_pred = self.predict_every_model(X)
        submission = self.combine_result(y_pred, Id)
        return submission


if __name__ == "__main__":

    # predict for test data
    param = {
        "kfold": 5,
        "Id": "id",
        "drop_cols": "id",
        "feature_pipeline_pkl": "pipeline_X.pkl",
        "test_data": "input/test_folds.csv",
    }

    # for start_date,end_date
    submission = RegressionPredict(**param).final_predict()
    submission.to_csv("model_preds/submission.csv", index=False)
