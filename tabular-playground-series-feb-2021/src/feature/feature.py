import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.base import TransformerMixin, BaseEstimator

seed = 42


class CustomGMM(TransformerMixin, BaseEstimator):

    """
    Gaussian mixture model

    component column names and number of component
    """

    def __init__(self, col_suffix, cols_component, predict_proba, **kwargs):
        self.cols_component = cols_component
        self.col_suffix = col_suffix
        self.predict_proba = predict_proba

    def fit(self, X, y=None):
        """
        fit data to model
        """
        self.gmm = {}
        for col, component in self.cols_component.items():
            gmm = GaussianMixture(n_components=component, random_state=seed)
            val = X[col].values.reshape(-1, 1)
            gmm.fit(val)
            self.gmm[col] = gmm
        return self

    def transform(self, X):
        for col, component in self.cols_component.items():
            val = X[col].values.reshape(-1, 1)
            if self.predict_proba:
                # predict probability
                proba = self.gmm[col].predict_proba(val)
                proba = proba[:, :-1]

                # concat data to original frame
                col = col + self.col_suffix + "_"
                cols = [col + f"{w}" for w in range(proba.shape[1])]
                proba = pd.DataFrame(proba, columns=cols)
                X = pd.concat([X, proba], axis=1)
            else:
                X[col + self.col_suffix] = self.gmm[col].predict(val)
        return X
