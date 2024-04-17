from modules.preprocessing import PreProcessing
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np


class PolyPreProcessing(PreProcessing):
    """
    A wrapper for SciKit Learn Polynomial Features.

    ref: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html

    :param columns: columns to be considered in the polynomial
    :param kwargs: keyword args for PolynomialFeatures transform.
    """
    def __init__(self, columns, **kwargs):
        super().__init__(columns)
        self.poly_maker = PolynomialFeatures(**kwargs)
        self.columns = columns

    def transform(self, df: pd.DataFrame):
        np_poly = self.poly_maker.fit_transform(
            np.nan_to_num(df[self.columns])
        )
        self.poly_features = self.poly_maker.get_feature_names_out(self.columns)  # noqa
        df_poly = pd.DataFrame(
            np_poly,
            columns='poly_' + self.poly_features,
            index=df.index
        )
        df_poly.columns = 'poly_' + df_poly.columns
        return pd.concat([df, df_poly], axis=1)
