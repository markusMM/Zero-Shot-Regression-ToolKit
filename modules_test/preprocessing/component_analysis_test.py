# import pytest
import numpy as np
import pandas as pd

from modules.preprocessing.component_analysis import (
    ICAPreProcessing,
    PolyPCAPreProcessing,
    RBFPCAPreProcessing,
    NMFPreProcessing,
    PCAPreProcessing,
    SigmoidPCAPreProcessing,
)


class TestDimRed:

    df = pd.DataFrame(np.random.rand(800, 40), columns=[f"c_{j}" for j in range(40)])

    ca_models = (
        ICAPreProcessing,
        PolyPCAPreProcessing,
        RBFPCAPreProcessing,
        NMFPreProcessing,
        PCAPreProcessing,
        SigmoidPCAPreProcessing,
    )

    def test_dim_red(self):

        for model in self.ca_models:
            x = model([f"c_{j}" for j in range(40)], n_components=20).transform(
                self.df.copy()
            )
            ca_cols = set(x.columns).difference(self.df.columns)
            assert len(ca_cols) == 20
            assert np.abs(x.iloc[:, -20:].corr().values.sum()) - 20 < 0.05
