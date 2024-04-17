import os
from distutils.dir_util import mkpath
import pandas as pd
import numpy as np
from modules.data_loading import generate_dataloader
from modules.common import ROOT


class TestGenerateDataLoader:

    df = pd.DataFrame(
        100 + 2*np.random.rand(400, 14),
        columns=[
            f'p_{j}' for j in range(4)
        ] + [
            f'v_{k}' for k in range(4, 14)
        ]
    )

    def test_from_df(self):
        df = self.df.copy()
        xcols = [f'v_{k}' for k in range(6, 12)]
        ycols = [f'p_{j}' for j in range(3)]
        batch_size = 12
        dl = generate_dataloader(
            df,
            ycols=ycols,
            xcols=xcols,
            batch_size=batch_size
        )
        for j, (x, y) in enumerate(dl):
            if j < len(df) // batch_size:
                ndim = x.shape[0]
                xdim = x.shape[1]
                ydim = y.shape[1]
                assert ndim == batch_size
                assert xdim == len(xcols)
                assert ydim == len(ycols)

    def test_from_file(self):
        df = self.df.copy()
        path = os.path.join(ROOT, 'data/tmp')
        mkpath(path)
        list(map(
            lambda b:
            df.iloc[
                b * 100:min(len(df), (b + 1) * 100)
            ].to_csv(os.path.join(path, f'test_csv_{b}.csv')),
            range(len(df) // 100 + 1 * (len(df) % 100))
        ))
        xcols = [f'v_{k}' for k in range(6, 12)]
        ycols = [f'p_{j}' for j in range(3)]
        batch_size = 12
        dl = generate_dataloader(
            path,
            ycols=ycols,
            xcols=xcols,
            batch_size=batch_size
        )
        for j, (x, y) in enumerate(dl):
            if j < len(df) // batch_size:
                ndim = x.shape[0]
                xdim = x.shape[1]
                ydim = y.shape[1]
                assert ndim == batch_size
                assert xdim == len(xcols)
                assert ydim == len(ycols)
