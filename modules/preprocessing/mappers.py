import pandas as pd

from modules.log import logger
from modules.preprocessing.pre_processing import PreProcessing
from modules.data_retrieval.df_handlers import parse_device_types


class Mapper(PreProcessing):
    """
    Map columns with a mapping function.

    Takes whatever is in the mapping dictionary for column *c* and applies the map
    function to that columns with it.

    Link: https://pandas.pydata.org/docs/reference/api/pandas.Series.map.html

    :param columns: columns to be processed
    :param maps: dictionary with mapping for each column
    """

    def __init__(
            self,
            columns: list,
            maps: dict
    ):
        super().__init__(columns)
        self.maps = maps

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for c in self.columns:
            if c not in self.maps:
                cmap = pd.Series(df[c].unique())
                idx = cmap.index
                cmap = {cmap[j]: j for j in idx}
            else:
                cmap = self.maps[c].copy()
            if isinstance(df[c][df[c] == df[c]].iloc[0], list):
                def apply_cmap(x): return [cmap[v] if v in cmap else None for v in x]
            else:
                def apply_cmap(x): return cmap[x]
            df[c] = df[c].map(apply_cmap)

        return df


class Aliasing(PreProcessing):
    """
    Alias columns.

    You secify `columns` and `aliases`.
    `aliases` has to be a dict with keys ~ columns and values being the new names!

    :param columns: list of columns to be processed.
    :param aliases: dictionary with keys ~ `columns` and values, the respective aliases.
    """

    def __init__(self, columns, aliases):
        super().__init__(columns)
        self.aliases = aliases

    def transform(self, df: pd.DataFrame):
        aliases = self.aliases
        for c in self.columns:
            if c in aliases:
                if (c in df) and (aliases[c] not in df):
                    df = df.rename(columns={c: aliases[c]})
                else:
                    logger.warn(f'cannot rename / find column {c}')

        return df


class ParseDeviceType(PreProcessing):
    """
    Alias some XANDR device types to our device types.

    It uses :method:`~modules.data_retrieval.df_handlers.parse_device_type`

    :param columns: columns which contain the lists of device types
    """

    def __init__(
            self,
            columns: list = ['device_type']
    ):
        super().__init__(columns)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for c in self.columns:
            df = parse_device_types(df, c)

        return df
