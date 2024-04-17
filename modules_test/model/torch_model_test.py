from distutils.dir_util import mkpath
import pandas as pd
import numpy as np
from torch.distributions import Bernoulli
from modules.loss_functions.torch_outlier_detectors import std_threshold
from modules.model.torch_model import LightningModel
from modules.model.algos.neural_net import MLP


class TestLightningModel:

    df = pd.DataFrame(
        100 + 2*np.random.rand(400, 14),
        columns=[f'p_{j}' for j in range(4)]+[f'v_{k}' for k in range(4, 14)]
    )

    def test_fit_predict(self):

        model = LightningModel(
            feature_columns=[f'v_{k}' for k in range(4, 9)],
            target_columns=[f'p_{j}' for j in range(1, 4)],
            model_class=MLP,
            model_args=dict(
                n_input=5,
                n_output=3,
                n_hidden=[10, 5],
                use_response_adbudg=True
            ),
            train_idx=np.arange(370),
            valid_idx=np.arange(370, 400),
            max_epochs=5
        ).fit(self.df)

        y = model.predict(self.df)

        assert y.shape[1] == 3
        assert isinstance(y, pd.DataFrame)
        assert y.columns.tolist() == [f'p_{j}' for j in range(1, 4)]

    def test_fit_predict_from_files(self):
        df = self.df.copy()
        mkpath("data/tmp")
        list(map(
            lambda b:
            df.iloc[
                b * 100:min(len(df), (b + 1) * 100)
            ].to_csv(f'data/tmp/test_csv_{b}.csv'),
            range(len(df) // 100 + 1 * (len(df) % 100))
        ))
        model = LightningModel(
            feature_columns=[f'v_{k}' for k in range(4, 9)],
            target_columns=[f'p_{j}' for j in range(1, 4)],
            model_class=MLP,
            model_args=dict(
                n_input=5,
                n_output=3,
                n_hidden=[10, 5],
                use_response_adbudg=True
            ),
            train_idx=np.arange(370),
            valid_idx=np.arange(370, 400),
            max_epochs=5
        ).fit("data/tmp")

        y = model.predict(self.df)

        assert y.shape[1] == 3
        assert isinstance(y, pd.DataFrame)
        assert y.columns.tolist() == [f'p_{j}' for j in range(1, 4)]

    def test_fit_predict_outliers(self):

        df = self.df.copy()
        xcols = [f'v_{k}' for k in range(4, 9)]
        x = df[xcols].values
        # paturbating with outliers
        pat = Bernoulli(.04).sample(df[xcols].shape)
        x[pat.bool().detach().numpy()] = 99999
        df[xcols] = x

        # fit
        model = LightningModel(
            feature_columns=xcols,
            target_columns=[f'p_{j}' for j in range(1, 4)],
            model_class=MLP,
            model_args=dict(
                n_input=5,
                n_output=3,
                n_hidden=[10, 5],
                use_response_adbudg=True,
                cut_outliers=True,
                out_fun=std_threshold
            ),
            train_idx=np.arange(370),
            valid_idx=np.arange(370, 400),
            max_epochs=3,
            batch_size=len(df)
        ).fit(df)

        y = model.predict(self.df)

        assert y.shape[1] == 3
        assert isinstance(y, pd.DataFrame)
        assert y.columns.tolist() == [f'p_{j}' for j in range(1, 4)]

        assert model.model.outliers.sum() > pat.mean(-1).sum()
        assert len(model.model.outliers) < len(pat)
