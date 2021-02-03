from sklearn import ensemble
from sklearn import neighbors
from sklearn import linear_model
import lightgbm as lgb
import xgboost as xgb

seed = 42
# =============================================================================
# MODELS
# =============================================================================

MODELS = {
    "lgbm": lgb.LGBMRegressor(
        objective="regression",
        boosting_type="gbdt",
        random_state=seed,
        n_jobs=-1,
        learning_rate=0.031473086821886716,
        colsample_bytree=0.6091584128089031,
        subsample=0.9241571230424561,
        min_split_gain=0.21976429840997636,
        min_child_samples=46,
        n_estimators=450,
        max_depth=12,
        num_leaves=1000,
        reg_alpha=0.6042849311204248,
        reg_lambda=0.5033218973570638,
    )
}


MODELS1 = {
    "lgbm": lgb.LGBMRegressor(
        objective="regression",
        boosting_type="gbdt",
        random_state=seed,
        n_jobs=-1,
        learning_rate=0.031473086821886716,
        colsample_bytree=0.6091584128089031,
        subsample=0.9241571230424561,
        min_split_gain=0.21976429840997636,
        min_child_samples=46,
        n_estimators=450,
        max_depth=12,
        num_leaves=1000,
        reg_alpha=0.6042849311204248,
        reg_lambda=0.5033218973570638,
    ),
    "lgbm": lgb.LGBMRegressor(
        objective="regression",
        boosting_type="gbdt",
        random_state=seed,
        n_jobs=-1,
        learning_rate=0.057522059597375975,
        colsample_bytree=0.602236296997475,
        subsample=0.6081685320944056,
        min_split_gain=0.21976429840997636,
        min_child_samples=78,
        n_estimators=824,
        max_depth=8,
        num_leaves=4400,
        reg_alpha=0.1307508343702849,
        reg_lambda=0.645428508278035,
    ),
}
