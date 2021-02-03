from sklearn.mixture import GaussianMixture
from sklearn.base import TransformerMixin, BaseEstimator

seed = 42


class CustomGMM(TransformerMixin, BaseEstimator):

    """
    Gaussian mixture model

    component column names and number of component
    """

    def __init__(self, col_suffix, cols_component, **kwargs):
        self.cols_component = cols_component
        self.col_suffix = col_suffix

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
            X[col + self.col_suffix] = self.gmm[col].predict(val)
        return X


