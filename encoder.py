from sklearn.base import BaseEstimator, TransformerMixin
from pandas.api.types import CategoricalDtype
import pandas as pd

class CatEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, min_count=1, dummy_na=False):
        self.min_count = min_count
        self.dummy_na = dummy_na
        self.categories = dict()
        self.cat_cols = None

    def fit(self, X):
        self.cat_cols = X.select_dtype(object).columns
        for cat_col in self.cat_cols:
            counts = pd.value_counts(X[cat_col])
            self.categories[cat_col] = counts[counts >= self.min_count].index.tolist()
        return self

    def transform(self, X):
        X = X.astype({cat_col: CategoricalDtype(self.categories[cat_col], ordered=True) 
                      for cat_col in self.cat_cols})
        transformed_X = pd.get_dummies(X, dummy_na=self.dummy_na)
        return transformed_X