import pytest
import pandas as pd
import numpy as np
from modules.model import ModelWrapper, SciKitRegressor
from modules.model.model_wrapper import FactorizedModel
from sklearn.linear_model import Lasso
from modules.preprocessing import CosPeriod
from modules.preprocessing.factorize import UniFactorize


class TestModelWrapper:

    df = pd.DataFrame(
        np.random.rand(800, 45),
        columns=[f"f_{j}" for j in range(40)] + [
            "viewed_imps",
            "clicks",
            "post_click_convs",
            "post_view_convs",
            "unique_imps"
        ]
    )
    model = ModelWrapper(
        [f"f_{j}" for j in range(40)],
        [
            "viewed_imps"
        ]
    )

    @pytest.mark.dependency(name="mod:fit",)
    def test_fit(self):
        model = self.model.fit(self.df)
        assert hasattr(model, "mu_x")
        self.model = model

    @pytest.mark.dependency(name="mod:pred", depends=["mod:fit"])
    def test_pred(self):

        y = self.model.predict(self.df)
        assert y.shape == (800, 1)
        assert y.columns == self.model.targets

    @pytest.mark.dependency(name="mod:conf", depends=["mod:fit"])
    def test_confidence_shape(self):
        y = self.model.predict(self.df)
        assert y.shape == (800, 1)
        assert np.isclose(
            self.model.mu_x,
            self.df[self.model.features].values.mean(axis=0)
        )

    @pytest.mark.dependency(name="mod:vartest", depends=["mod:fit"])
    def test_confidence_shape(self):
        vartest = self.model.predict_variance_test(self.df)
        assert 'feature2prediction_variance' in vartest
        assert 'target2prediction_variance' in vartest


class TestFactorizedModel:

    df = pd.DataFrame(
        np.random.rand(800, 45),
        columns=[f"f_{j}" for j in range(40)] + [
            "viewed_imps",
            "clicks",
            "post_click_convs",
            "post_view_convs",
            "unique_imps"
        ]
    )
    model = FactorizedModel(
        [f"f_{j}" for j in range(40)],
        [
            "viewed_imps"
        ],
        model_class=Lasso,
        model_params={},
        factor_models={
            'hour': CosPeriod(
                [f'f_{j}' for j in range(24)],
                16
            ),
            'f_x30': UniFactorize(
                [f'f_{j}' for j in range(24, 31)],
            ),
        }
    )

    # def test_multi_instance_array(self):

    #     df = self.df.copy()
    #     model = self.model.fit(df)
    #     pred = model.predict(df)
    #     assert df['viewed_imps'].shape == pred.values.squeeze().shape

    def test_multi_instance_dataframe(self):

        df = self.df.copy()
        model = self.model
        model.model = SciKitRegressor(
            model.features,
            model.targets,
            Lasso
        )
        model.xy_split = False
        model = model.fit(df)
        pred = model.predict(df)
        assert df['viewed_imps'].shape == pred.values.squeeze().shape
