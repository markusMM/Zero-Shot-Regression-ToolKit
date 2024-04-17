import numpy as np
import pandas as pd
from datetime import datetime as dt, timedelta as td
import multiprocessing

from modules.preprocessing.time import (
    GaussianDummy,
    ConTime,
    SeasonProcessing
)


class TestSeasonPreProcessing:
    ts = pd.Series(pd.date_range(dt.today(), dt.today() + td(days=720)).values)
    ts.name = 'date'
    df = pd.DataFrame(
        np.random.randn(len(ts), 10),
        columns=[f'c{j}' for j in range(10)]
    ).join(ts)

    def test_weekday_name(self):
        pp = SeasonProcessing(['date'], 'day', True)
        df = pp.transform(self.df)
        assert 'day' in df
        assert 'Monday' in df['day'].unique()

    def test_month(self):
        pp = SeasonProcessing(['date'], 'month', False)
        df = pp.transform(self.df)
        assert 'month' in df
        assert 10 in df['month'].unique()


class TestConTime:
    df = pd.DataFrame(np.stack([
        np.array(list(range(7))*2),
        np.random.randn(14),
        np.random.rand(14)
    ], axis=1), columns=[
        'ts',
        'rng_1',
        'rng_2'
    ])

    def test_contime(self):
        preproc = ConTime(['ts'], {'ts': 7})
        df1 = preproc.transform(self.df)
        preproc = ConTime(['ts'])
        df2 = preproc.transform(self.df)
        x = np.array(list(range(7))*2)
        y = np.sin(2 * np.pi * x / 7).tolist()
        assert 'ts_num_cont' in df1
        assert np.allclose(df1['ts_num_cont'], y)
        assert 'ts_num_cont' in df2
        assert np.allclose(df2['ts_num_cont'], y)


class TestGaussianDummy:
    df = pd.DataFrame(
        np.concatenate([
            pd.get_dummies(
                np.random.randint(0, 7, (80_000,))
            ).values,
            np.random.rand(80_000, 1)
        ], axis=1),
        columns=[
            f'c_{k}' for k in range(7)
        ] + ['beta']
    )

    def test_gaudum(self):
        t = dt.now().second
        preproc = GaussianDummy(
            [f'c_{k}' for k in range(7)],
            list(range(7))
        )
        df = preproc.transform(self.df)
        for c in pd.Series([f'c_{k}' for k in range(7)]) + '_gau':
            assert c in df
        assert 'beta' in df
        assert (dt.now().second - t) < 2.25*16/multiprocessing.cpu_count()
