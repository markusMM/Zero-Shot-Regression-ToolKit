import pandas as pd
import numpy as np
from modules.preprocessing.pre_processing import PreProcessing
from typing import Optional, Dict, List
from .factorize import *  # noqa


class DummyPreProcessing(PreProcessing):
    def __init__(self, columns, items: Optional[Dict[str, List]] = None):
        """
        Dummy categorical.

        `columns` get scanned for either items (key=name, value=list of items) or all
        uniques from all lists.

        The output are K columns, 0 or 1, for each instance.
        K is the number of unique values found at the original column.

        :param columns: columns to preprocess.
        :param items: dictionary (key=name, value=items), specifying what to scan for
        """
        super().__init__(columns)
        if not isinstance(items, dict):
            items = {}
        self.items = items

    def transform(self, df) -> pd.DataFrame:

        dummies = []
        for col in self.columns:
            u = self.items.get(col, None)
            if u is None:
                u = df[col].dropna().tolist()
                u = pd.Series(pd.Series(u).dropna().unique()).sort_values().values
                self.items[col] = u
            for k in u:
                ser = df[col].copy() == k
                ser.name = f'{col}_{k}'
                dummies.append(ser)

        dummies = pd.concat(dummies, axis=1)
        df = df[list(set(df.columns).difference(dummies.columns))]
        return df.join(dummies)


class DummyListPreProcessing(PreProcessing):
    def __init__(self, columns, items: Optional[Dict[str, List]] = None):
        """
        Dummy List Preprocessing.

        Input are `columns` of lists with arbitrary values.

        They either got some specified `items` (key=name, value=items) or get scanned
        for every unique entry from all lists of the specified column.

        Output are K columns per input columns, with 1 or 0, for each instance.
        K is the number of unique values found at the original column.

        :param columns: columns to preprocess.
        :param items: dictionary (key=name, value=items), specifying what to scan for
        """
        super().__init__(columns)
        if not isinstance(items, dict):
            items = {}
        self.items = items

    def binarize_list(self, list_x, col) -> pd.DataFrame:
        def filter_list(li, k):
            if li is None:
                return False
            if li == np.NaN:
                return False
            if not isinstance(li, list):
                return li == k
            return k in li

        u = self.items.get(col, None)
        if u is None:
            u = np.concatenate(list_x.dropna().tolist())
            u = pd.Series(pd.Series(u).dropna().unique()).sort_values().values
            self.items[col] = u
        df = []
        for k in u:
            ser = list_x.apply(lambda x: filter_list(x, k))
            ser.name = f'{col}_{k}'
            df.append(ser)
        return pd.concat(df, axis=1)

    def transform(self, df) -> pd.DataFrame:
        for c in self.columns:
            dummy_list = []
            dummies = self.binarize_list(df[c], c)
            dummy_list.append(dummies)
            dummies = pd.concat(dummy_list, axis=1)
            df = df[list(set(df.columns).difference(dummies.columns))]
            df = pd.concat([df, dummies], axis=1)
        return df


class AutoFillEmpty(PreProcessing):

    def __init__(
        self,
        columns: list
    ):
        """
        Autofill missing boolean targets.

        For those values in `columns` which are missing, activate all of them (x=1).

        NOTE: `columns` here is just one binary encoded feature.

        :param columns: columns to be used.
        """
        super().__init__(columns)

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        ''''pass'''
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df.loc[df[self.columns].values.sum(1) == 0, self.columns] = 1
        return df
