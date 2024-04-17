import pandas as pd
import numpy as np
from modules.preprocessing.bundle import ConCatEmbeddings

crt1 = 'abd'
crt2 = 'kls_o'
crt3 = 'ddd_9_a'
crt4 = '_f_7_'
cols1 = [f'{crt1}_{j}' for j in range(4)]
cols2 = [f'{crt2}_{j}' for j in range(4)]
cols3 = [f'{crt3}_{j}' for j in range(5)]
cols4 = [f'{crt4}_{j}' for j in range(5)]
df = pd.concat([
    pd.DataFrame(
        np.random.randn(100, 8),
        columns=cols1 + cols2
    ),
    pd.DataFrame(
        np.random.randint(1, 4, [100, 5]),
        columns=cols3
    ),
    pd.DataFrame(
        np.random.randint(7, 30, [100, 5]).astype(str),
        columns=cols4
    )
], axis=1)


class TestConCatEmbeddings:

    def test_without_default(self):
        pp = ConCatEmbeddings(columns=cols1)
        dfx = pp.transform(df.copy())
        dcol = list(set(dfx.columns).difference(df.columns))[0]
        assert dcol == crt1

    def test_with_default(self):
        pp = ConCatEmbeddings(columns=cols1, new_col='peter')
        dfx = pp.transform(df.copy())
        dcol = list(set(dfx.columns).difference(df.columns))[0]
        assert dcol == 'peter'

    def test_mixed_nodef(self):
        pp = ConCatEmbeddings(columns=cols1 + cols2)
        dfx = pp.transform(df.copy())
        dcol = list(set(dfx.columns).difference(df.columns))[0]
        assert dcol == cols1[0]

    def test_mixed_nodef(self):
        pp = ConCatEmbeddings(columns=cols1 + cols2, new_col='Hans')
        dfx = pp.transform(df.copy())
        dcol = list(set(dfx.columns).difference(df.columns))[0]
        assert dcol == 'Hans'
