import pytest

from modules.model.model_grouper import ModelGrouper
from sklearn.ensemble import RandomForestRegressor as RFR
from modules.model.scikit_model import SciKitRegressor
import numpy as np
import pandas as pd


class TestModelGrouper:
    group = np.random.randint(1, 11, 1000)
    values = np.random.randn(1000, 80)
    data = np.hstack([group[:, None], values])
    df = pd.DataFrame(data, columns=['g'] + [f'p_{j}' for j in range(80)])
    model = None

    @pytest.mark.dependency(
        name='model_grouper:fit'
    )
    def test_fit_predict_score(self):
        g_model = ModelGrouper(
            features=[f'p_{j}' for j in range(4, 10)],
            targets=[f'p_{k}' for k in range(4)],
            grouper='g',
            model_class=SciKitRegressor,
            model_args={'model_class': RFR}
        )
        g_model = g_model.fit(self.df)
        assert len(g_model.model) == 10
        self.model = g_model
        pred = self.model.predict(self.df)
        assert pred.shape[1] == 4
        assert pred.columns.tolist() == [f'p_{k}' for k in range(4)]
        scores = self.model.scores(self.df)
        assert scores.shape == (10, 4)
        score = self.model.score(self.df)
        assert score <= 1.0
