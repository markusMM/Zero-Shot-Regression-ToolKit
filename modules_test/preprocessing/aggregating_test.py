import os
import pandas as pd
import numpy as np
from modules.preprocessing.aggregating import Aggregator
from modules.common import ROOT as HERE
TEST_DATA_PATH = os.path.join(
    HERE,
    "modules_test/test_data/tr_data.csv"
)


class TestAggregator:

    df = pd.read_csv(
        TEST_DATA_PATH,
        index_col=0
    )

    def test_log_case(self):

        pp = Aggregator(
            columns=[
                'f_00',
                'log_f_00',
                'f_03',
                'log_f_03',
                'f_02',
                'log_f_02',
                'f_01',
                'log_f_01'
            ],
            indices=['id_01'],
            col_agg={
                'f_00': np.nansum,
                'log_f_00': np.nanmean,
                'f_01': np.nansum,
                'log_f_01': np.nanmean
            }
        )
        df_pp = pp.transform(self.df.copy())

        for c in ['f_00', 'f_01']:
            assert (
                df_pp[
                    f'log_{c}'
                ].values != np.log1p(self.df.groupby(
                    'id_01'
                )[c].agg(np.nanmean)).values
            ).sum() < 10
            assert (
                df_pp[
                    c
                ].values != self.df.groupby(
                    'id_01'
                )[c].agg(np.nansum).values
            ).sum() < 10
