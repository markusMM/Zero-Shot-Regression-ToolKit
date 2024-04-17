import pandas as pd
import numpy as np
from modules.hpo.tuning_utils import predict_variance_test
from modules.model import ModelWrapper
N = 200
D = len(list(range(11, 32)))
K = 4
DATA = pd.DataFrame(
    np.random.rand(N, 40),
    columns=[f'c_{j}' for j in range(40)]
)
TARGET = pd.DataFrame(
    100 + 10*np.random.rand(N, K) + 1.2*np.random.randn(N, K),
    columns=['a', 'b', 'c', 'x']
)
DATA = pd.concat([DATA, TARGET], axis=1)
MODEL = ModelWrapper(
    feature_columns=[f'c_{k}' for k in range(11, 32)],
    target_columns=['a', 'b', 'c', 'x']
)


class TestPredictVarianceTest:
    def test_predict_variance_test(self):
        var_test = predict_variance_test(MODEL, DATA)
        assert 'feature2prediction_variance' in var_test
        assert 'target2prediction_variance' in var_test
