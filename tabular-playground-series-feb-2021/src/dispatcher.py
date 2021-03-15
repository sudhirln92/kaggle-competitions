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
        n_jobs=-1,
        n_estimators=3000,
        learning_rate=0.02,
        random_state=seed,
        **{
            "cat_smooth": 7.530798694305587,
            "colsample_bytree": 0.5947095495005181,
            "max_depth": 72,
            "min_child_samples": 70,
            "min_split_gain": 0.0011369738838037113,
            "num_leaves": 122,
            "reg_alpha": 2.3886192425757202,
            "reg_lambda": 0.008028993252510504,
            "subsample": 0.7837832722604414,
        }
    )
}


MODELS1 = {
    "xgbm": xgb.XGBRegressor(
        objective="reg:linear",
        random_state=seed,
        n_jobs=-1,
        learning_rate=0.03811393134007876,
        colsample_bytree=0.6862274468156859,
        subsample=0.7465222086484357,
        n_estimators=987,
        max_depth=6,
    ),
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
