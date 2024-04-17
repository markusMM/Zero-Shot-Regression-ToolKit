import re
import multiprocessing

import pandas as pd, numpy as np
from modules.data_retrieval.aurora_harvest import get_aurora_query
from modules.data_retrieval.df_handlers import save_unstring_elem
from modules.preprocessing.pre_processing import PreProcessing
from modules.common import NCPU


class DecodeRegion(PreProcessing):
    """
    Decode Region.

    Translates columns of lists with geo-targeting IDs to columns of lists of their
    respective written names. This helps to convert them into contextual embeddings.
    This method here assumes that each entry within the lists starts with a two
    character country code (e.g. 'DE' for Germany) and then the specifed ID.

    :param sub: a substring to forc the data columns for to be processed.
    :param reftab: a dataframe containing all specified IDs and country codes.
    :param val_cols: the columns to be concatenated for the final names.
    """
    def __init__(
            self,
            sub: str = '_region',
            reftab: pd.DataFrame = None,
            val_cols: list = ['name', 'country_code']
    ):
        super().__init__([])
        if reftab is None:
            reftab = get_aurora_query('select * from appnexus.region;')
        self.reftab = reftab
        self.sub = sub
        self.val_cols = val_cols

    def transform(self, df: pd.DataFrame):
        """
        Transform all elements inside all entries of all given columns with a re.sub.

        The replacement is done by the `trans_elem` function.
        Ideally, the output columns should all be object type containing ready to use lists!

        :param df:
        :return: dataframe with additional columns with lists of names (indead of IDs).
        """
        # parse columns to be transformed
        if isinstance(self.sub, list) or self.sub in df:
            region_cols = self.sub
        else:
            region_cols = [col for col in df.columns if self.sub in col]
        if isinstance(region_cols, str):
            region_cols = [region_cols]

        # merge df with transformed columns
        return pd.concat([df] + [self.trans_col(df[col]) for col in region_cols], axis=1)

    def trans_col(self, col: pd.Series) -> pd.Series:
        name = col.name
        col = col.astype(str).str.replace(
            "'null',",
            '',
            regex=False
        )
        with multiprocessing.Pool(NCPU) as pool:
            col = pd.Series(pool.map(
                self.trans_entry,
                col.values
            ), name=name + '_name')

        return col.apply(save_unstring_elem)

    def trans_entry(self, entry: str) -> list:
        return save_unstring_elem(re.sub(
            "'[A-Z][A-Z]\\d+'",
            self.trans_elem,
            entry
        ))

    def trans_elem(self, x: str) -> str:
        x = x.group(0)
        x = x.replace('\'', '').replace('"', '')
        if x == 'null':
            return "None"
        if len(x) < 3:
            return "None"
        if not isnum(x[2:]):
            return "None"
        y = self.reftab.loc[
            (self.reftab['country_code'] == x[:2]) &
            (self.reftab['id'] == int(x[2:])),
            self.val_cols
        ]
        if len(y) < 1:
            return "None"
        if len(y) > 1:
            y = y.iloc[0]
        y = ', '.join(y.values.squeeze().tolist())
        if len(y) < 1:
            return "None"
        return f"'{y}'"


class DecodeTrackingFromIDs(PreProcessing):
    """
    Decode tracking Info from IDs.

    Translates columns of lists with tracking IDs to columns of lists of their
    respective written names. This helps to convert them into contextual embeddings.
    Additionally, an unwanted regex patter `val_re_strip` can be stripped from each
    item, providing this parameter.

    :param sub: a substring to forc the data columns for to be processed
    :param reftab: a dataframe containing all specified IDs and country codes
    :param lit_col: The column to be extracted from reftab dataframe
    :param val_re_strip: regex to be stripped from each item
    """
    def __init__(
            self,
            sub: str,
            reftab: pd.DataFrame,
            lit_col: str = 'name',
            val_re_strip: str = None
    ):
        super().__init__([])
        self.reftab = reftab
        self.sub = sub
        self.lit_col = lit_col
        self.val_re_strip = val_re_strip

    def transform(self, df: pd.DataFrame):
        for pc_col in filter(lambda c: self.sub in c, df.columns):
            df[pc_col + '_' + self.lit_col] = df[pc_col].apply(
                lambda y:
                    pd.Series(list(map(
                        lambda x:
                            self.trans_elem(x),  # noqs
                        y
                    ))).dropna().unique().tolist()
            )
        return df

    def trans_elem(self, x: str) -> str:

        # excluding special / missing case
        if x in ['--', 'null', None, np.nan] or not isnum(x):
            return None

        # searching for 'id'
        r_id = np.where((
            self.reftab['id'] == int(x)
        ).values)[0]
        if len(r_id) < 1:
            return None

        # mapping to first hit
        r_id = r_id[0]
        x = self.reftab.iloc[r_id][
            self.lit_col
        ]

        # value re strip
        if self.val_re_strip is not None:
            x = re.sub(self.val_re_strip, '', x).strip()

        return x


def isnum(s: str):
    try:
        s = int(s)
        return True
    except Exception:  # noqa
        return False
