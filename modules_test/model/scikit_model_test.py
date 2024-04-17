import pytest
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from modules.model.scikit_model import SciKitRegressor


class TestSciKitRegressor:

    df = pd.DataFrame(
        np.random.rand(800, 42),
        columns=[f"f_{j}" for j in range(40)] + [f"t_{j}" for j in range(2)],
    )
    model = SciKitRegressor(
        [f"f_{j}" for j in range(40)],
        [f"t_{j}" for j in range(1)],
        SVR,
        kernel="rbf",
        gamma=0.9,
    )

    @pytest.mark.dependency(name="mod:fit",)
    def test_fit(self):
        model = self.model.fit(self.df)
        assert hasattr(model.model, "support_vectors_")
        self.model = model

    @pytest.mark.dependency(name="mod:pred", depends=["mod:fit"])
    def test_pred(self):

        y = self.model.predict(self.df)
        assert y.shape == (800, 1)
        assert y.columns == self.model.targets
