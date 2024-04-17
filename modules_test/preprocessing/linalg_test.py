import numpy as np
import pandas as pd
from modules.preprocessing.linalg import LowRankPCACompression
from modules.log import logger


class TestlowRankPCACompression:
    ex_df = pd.DataFrame(
        np.random.randn(800, 612),
        columns=[
            f'd_{j}' for j in range(20)
        ] + [
            f'x_{k}' for k in range(80)
        ] + [
            f'g_embed_{c}' for c in range(512)
        ]
    )

    def test_fit_transform(self):

        pre_proc = LowRankPCACompression(['g'], embed_cols=True)
        df1 = pre_proc.transform(self.ex_df.copy())
        logger.info(df1.columns)
        assert df1.shape[0] == self.ex_df.shape[0]
        assert df1.shape[1] > self.ex_df.shape[1]
        assert pre_proc.columns == [
            f'g_embed_{c}' for c in range(512)
        ]
        assert pre_proc.fitted
        df2 = pre_proc.transform(self.ex_df.copy())
        m1 = df1[[f'g_embed_pc_{n}' for n in range(pre_proc.n_comp_fit)]].values
        m2 = df2[[f'g_embed_pc_{n}' for n in range(pre_proc.n_comp_fit)]].values
        assert np.sqrt(
            ((m1 - m2)**2).mean()
        ) < .5*np.sqrt((np.abs(m1) + np.abs(m2)).max()**2 / 2)

        df3 = pre_proc.transform(self.ex_df.iloc[-3:])
        m3 = df3[[f'g_embed_pc_{n}' for n in range(pre_proc.n_comp_fit)]].values
        assert np.sqrt(((m1[-3:] - m3)**2 + (m2[-3:] - m3)**2).mean()) / 2 < 2
