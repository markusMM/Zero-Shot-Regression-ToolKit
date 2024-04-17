import re
import pandas as pd
import numpy as np

from modules.log import logger
from modules.preprocessing.pre_processing import PreProcessing
from modules.preprocessing.util import firstel


class Aggregator(PreProcessing):

    def __init__(
            self,
            columns: list = [],
            indices: list = ['creative_id'],
            dtype_agg: dict = dict(
                bool=np.nanmean,
                list=firstel,
                object=firstel,
                general=np.nanmean
            ),
            col_agg: dict = None
    ):
        """
        Aggregator.

        Uses two dicts with keys as identifier and value as aggregation function:
         - `col_agg` for column specific aggregations (overwriting the dtype one)
         - `dtype_agg` for data type specific aggregations

        The whole aggregation is done with `indices` as groupers.
        `columns` are the columns to keep, if empty all columns are tried to be kept.

        :param columns: columns to be kept, if not empty.
        :param indices: groupers / indices the aggregation resolution is done on
        :param dtype_agg: dict with key=dtype, value=aggregation function
        :param col_agg: dict with key=column name, value=aggregation function
        """
        super().__init__(columns)
        # ignoring numbers, like bit length
        dtype_agg = {
            re.sub('\d+', '', t): dtype_agg[t]  # noqa
            for t in dtype_agg.keys()
        }
        if 'general' not in dtype_agg:
            dtype_agg['general'] = np.mean
        self.indices = indices
        self.dtype_agg = dtype_agg
        self.col_agg = col_agg if col_agg is not None else {}
        self.c_agg = {}

    def transform(self, df: pd.DataFrame):
        if self.columns is None:
            self.columns = []
        if len(self.columns) <= 0:
            self.columns = set(df.columns).difference(self.indices)

        c_agg = {}
        log_cols = []
        for c in df.columns:
            # checking log case
            if len(re.findall('^log_', c)) > 0 and re.sub('^log_', '', c) in df:  # noqa
                df[c] = df[re.sub('^log_', '', c)]
                log_cols += [c]
            # prechecking if a special function is declared
            if c in getattr(self, 'col_agg', {}):
                agg = self.col_agg[c]
            else:
                # ignoring numbers, like bit lengths
                ctyp = re.sub('\d+', '', str(df[c].dtype))  # noqa
                if ctyp in self.dtype_agg:
                    agg = self.dtype_agg.get(str(df[c].dtype), self.dtype_agg['general'])
                else:
                    agg = self.dtype_agg['general']
            try:
                agg(df[c])
            except Exception as e:  # noqa
                logger.warn(e)
                logger.warn(
                    f'Cannot aggregate {c} with function {agg}! Using first element!'
                )
                agg = firstel
            c_agg[c] = agg

        df = df.groupby(self.indices).aggregate(c_agg)
        for lc in log_cols:
            df[lc] = np.log1p(df[lc])
        for ix in self.indices:
            if ix in df.columns:
                df = df.drop(ix, axis=1)
        self.c_agg = c_agg
        return df.reset_index(drop=False)
