# =============================================================================
# Import library
# =============================================================================
from sklearn import ensemble
from sklearn import neighbors
from sklearn import linear_model
import lightgbm as lgb
import xgboost as xgb
import catboost as cgb

from .utils.file_handler import read_config

config = read_config("config.json")
seed = config["seed"]
# =============================================================================
# MODELS
# =============================================================================

MODELS = {
    "lgbm": lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        n_estimators=3000,
        learning_rate=0.02,
        random_state=seed,
        n_jobs=-1,
        **{
            "num_leaves": 153,
            "max_depth": 14,
            "max_delta_step": 9,
            "reg_alpha": 14.206069641010822,
            "reg_lambda": 4.35151505977074,
            "colsample_bytree": 0.23599717695150987,
            "cat_smooth": 49.698724437071206,
            "cat_l2": 19,
        }
    )
}