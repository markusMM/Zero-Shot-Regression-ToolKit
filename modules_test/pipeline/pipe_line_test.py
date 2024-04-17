import os
import joblib
import pytest
import pandas as pd
import numpy as np
from scipy.stats import normaltest
import glob

from modules.preprocessing.scaling import (
    RobustScalerPreProcessing,
    NormalQuantilePreProcessing,
    UniformQuantilePreProcessing,
)
from modules.pipeline import PipeLine
from sklearn.linear_model import SGDRegressor
from modules.model.scikit_model import SciKitRegressor
from modules.model import ModelWrapper
from modules.common import ROOT

MODEL_PATH = os.path.join(ROOT, "modules_test/test_model/")


class TestLoadSave:

    model = PipeLine(
        preprocessings_a=[],
        model_a=ModelWrapper(
            ["f_05", "f_70", "f_93"],
            ["y_00", "y_01"]
        ),
        features_a=["f_05", "f_70", "f_93"],
        targets=["y_00", "y_01"],
    )

    @pytest.mark.dependency(name="pipe:save")
    def test_save(self):
        self.model.save(MODEL_PATH + "/temp_model.pkl")
        assert len(glob.glob(MODEL_PATH + "/temp_model.pkl")) > 0

    @pytest.mark.dependency(name="pipe:load", depends=["pipe:save"])
    def test_load(self):
        pipe = joblib.load(MODEL_PATH + "/temp_model.pkl")
        assert pipe.features_a == ["f_05", "f_70", "f_93"]
        assert pipe.targets == ["y_00", "y_01"]
        assert pipe.model_a.features == ["f_05", "f_70", "f_93"]
        assert pipe.model_a.targets == ["y_00", "y_01"]
        os.remove(MODEL_PATH + "/temp_model.pkl")


class TestPipeLine:

    df = pd.DataFrame(
        np.random.randn(800, 6),
        columns=["cpc", "cvr", "cpm", "age_1", "age_0", "budget"]
    )
    df['creative_id'] = np.random.randint(0, 21, 800)
    df['log_imps_viewed'] = 1.26 * np.random.randn(800)
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
    target_columns = ["cvr"]
    model = SciKitRegressor(
        feature_columns, target_columns, SGDRegressor, alpha=0.0003
    )

    @pytest.mark.dependency(name="pipe:pipe")
    def test_pipe(self):
        self.pipe = PipeLine(
            preprocessings_a=self.prep,
            model_a=self.model,
            features_a=self.feature_columns,
            targets=self.target_columns,
            model_b=self.model
        )
        assert self.pipe is not None

    @pytest.mark.dependency(name="pipe:preprocess", depends=["pipe:pipe"])
    def test_preprocess_data(self):
        self.pipe = PipeLine(
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
            self.pipe = PipeLine(
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

    @pytest.mark.dependency(name="pipe:model", depends=["pipe:preprocess"])
    def test_log_model(self):

        if not hasattr(self, "pipe"):
            self.pipe = PipeLine(
                preprocessings_a=self.prep,
                model_a=self.model,
                features_a=self.feature_columns,
                targets=self.target_columns,
                model_b=self.model
            )

        # self.pipe = self.pipe.fit(self.df)
        z = self.pipe.predict(self.df_use_case)
        assert 0 < len(z.iloc[:, 0]) < 2
        assert 0 < len(z.iloc[:, 1]) < 2

    # @pytest.mark.dependency(name="pipe:reco", depends=["pipe:model"])
    # def test_rec(self):
    #     self.pipe = PipeLine(
    #         preprocessings_a=self.prep,
    #         model_a=self.model,
    #         features_a=self.feature_columns,
    #         targets=self.target_columns,
    #         model_b=self.model
    #     ).fit(self.df)
    #     y = self.pipe.predict(self.df)['cvr'].values
    #     x_budget = self.pipe.recommend(self.df, 'budget', 'cvr')
    #     df = self.df.copy()
    #     df['budget'] = x_budget
    #     y_budgeted = self.pipe.predict(df)['cvr'].values

    #     assert np.nansum(y) <= np.nansum(y_budgeted)
