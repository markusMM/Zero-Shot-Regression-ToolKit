import pytest

from modules.training import grouped_training
from sklearn.ensemble import RandomForestRegressor as RFR
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


class TestGroupedTraining:
    group = np.random.randint(1, 11, 1000)
    values = np.random.randn(1000, 80)
    data = np.hstack([group[:, None], values])
    df = pd.DataFrame(data, columns=['g'] + [f'p_{j}' for j in range(80)])
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_ids, valid_ids = list(sss.split(df[[f'p_{j}' for j in range(4)]], df['g']))[0]

    @pytest.mark.dependency(
        name='model_grouper:fit'
    )
    def test_fit_predict_score(self):
        models, tr_scores, va_scores = grouped_training(
            df=self.df,
            xcols=[f'p_{j}' for j in range(4, 10)],
            ycols=[f'p_{k}' for k in range(4)],
            train_ids=self.train_ids,
            valid_ids=self.valid_ids,
            groupers=['g'],
            model_class=RFR
        )
        assert len(models) == 10
        assert len(tr_scores) == 10
        assert len(va_scores) == 10
        assert sum(map(lambda s: s > 1, tr_scores.values())) <= 0
        assert sum(map(lambda s: s > 1, va_scores.values())) <= 0
