import pytest
import pandas as pd
import numpy as np
from scipy.stats import normaltest

from modules.preprocessing.scaling import (
    RobustScalerPreProcessing,
    NormalQuantilePreProcessing,
    UniformQuantilePreProcessing,
)
from modules.pipeline.skopt_rec import SkOptRecPipe
from modules.model import SciKitRegressor
from sklearn.linear_model import SGDRegressor


class TestSkOptRecPipe:

    df = pd.DataFrame(
        np.random.randn(800, 6),
        columns=["cpc", "cvr", "cpm", "age_1", "age_0", "budget"]
    )
    df['creative_id'] = np.random.randint(0, 21, 800)
    df_use_case = pd.DataFrame(
        np.random.randn(1, 6),
        columns=["cpc", "cvr", "cpm", "age_1", "age_0", "budget"]
    )
    df_use_case['creative_id'] = 0

    prep = [
        NormalQuantilePreProcessing(["age_0", "age_1"]),
        RobustScalerPreProcessing(["budget"]),
        UniformQuantilePreProcessing(["cpm"]),
    ]
    feature_columns = ["budget", "age_0", "age_1", "cpm"]
    target_columns = ["cpc", "cvr"]
    model = SciKitRegressor(
        feature_columns, target_columns, SGDRegressor, alpha=0.003
    )

    @pytest.mark.dependency(name="pipe:pipe")
    def test_pipe(self):
        self.pipe = SkOptRecPipe(
            preprocessings_a=self.prep,
            model_a=self.model,
            features_a=self.feature_columns,
            targets=self.target_columns,
            model_b=self.model
        )
        assert self.pipe is not None

    @pytest.mark.dependency(name="pipe:preprocess", depends=["pipe:pipe"])
    def test_preprocess_data(self):
        self.pipe = SkOptRecPipe(
            preprocessings_a=self.prep,
            model_a=self.model,
            features_a=self.feature_columns,
            targets=self.target_columns,
            model_b=self.model
        )
        df10 = self.df
        df11 = self.pipe.preprocess_data(df10.copy())
        assert np.abs(
            (0 - df11['budget'].mean())
        ) < 0.05 * np.abs(df10['budget']).max()
        assert np.mean(normaltest(df11[["age_0", "age_1"]]).pvalue) < 0.05
        df20 = self.df_use_case
        df21 = self.pipe.preprocess_data(df20.copy())
        assert np.nansum(df21.values == df20.values) < .5*np.prod(df21.shape)

    @pytest.mark.dependency(name="pipe:model", depends=["pipe:preprocess"])
    def test_model(self):

        if not hasattr(self, "pipe"):
            self.pipe = SkOptRecPipe(
                preprocessings_a=self.prep,
                model_a=self.model,
                features_a=self.feature_columns,
                targets=self.target_columns,
                model_b=self.model
            )

        self.pipe = self.pipe.fit(self.df)
        z = self.pipe.predict(self.df_use_case)
        assert 0 < len(z.iloc[:, 0]) < 2
        assert 0 < len(z.iloc[:, 1]) < 2

    # @pytest.mark.dependency(name="pipe:reco", depends=["pipe:model"])
    # def test_rec(self):

    #     if not hasattr(self, "pipe"):
    #         self.pipe = SkOptRecPipe(
    #             preprocessings_a=self.prep,
    #             model_a=self.model,
    #             features_a=self.feature_columns,
    #             targets=self.target_columns,
    #             model_b=self.model
    #         ).fit(self.df.fillna(0))

    #     yn = self.pipe.predict(self.df)[self.target_columns]

    #     self.pipe = self.pipe.fit(self.df)
    #     xr = self.df.copy()
    #     rr = self.pipe.recommend(self.df_use_case, ["cpm"], "cvr", True)
    #     xr[rr.columns] = rr
    #     yr = self.pipe.predict(xr)[self.target_columns]

    #     assert yr.values.mean() >= yn.values.mean()
