from typing import List
import pandas as pd


class PreProcessing:
    """
    PreProcessing.

    A parent class for any preprocessing.

    It always takes a dataframe in and a dataframe out, with new / preprocessed columns.
    `columns` are to be preprocessed.

    :param columns: to be preprocessed
    :param kwargs: additional keyword args. (relevant for some children)
    """
    def __init__(self, columns: List, **kwargs):
        self.columns = columns

    def transform(self, df: pd.DataFrame):
        # do some processing of self.columns here
        return df
