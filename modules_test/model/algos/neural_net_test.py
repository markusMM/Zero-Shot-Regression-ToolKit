import pandas as pd
import numpy as np
import torch
from modules.model.algos.neural_net import MLP, SkipNet


class TestNeuralNet:

    df = pd.DataFrame(
        np.random.rand(800, 14),
        columns=[f'p_{j}' for j in range(4)]+[f'v_{k}' for k in range(4, 14)]
    )

    def test_forward(self):
        model = MLP(10, 4, [10, 5])
        y = model(torch.tensor(
            self.df[[f'v_{k}' for k in range(4, 14)]].values
        ).float())
        assert y.shape[1] == 4

    def test_sigmoid(self):
        model = MLP(
            10, 4, [10, 5],
            use_response_sigmoid=True
        )
        y = model(torch.tensor(
            self.df[[f'v_{k}' for k in range(4, 14)]].values
        ).float())
        assert y.shape[1] == 4
        h = torch.tensor(
            self.df[[f'v_{k}' for k in range(4, 14)]].values
        ).float()
        n = len(model.layers)
        with torch.no_grad():
            for j, l in enumerate(model.layers):
                if j == n - 2:
                    break
                h = l(h)
        h = h.detach().numpy()
        assert h.max() <= 1
        assert h.min() >= 0

    def test_adbudg(self):
        model = MLP(
            10, 4, [10, 5],
            use_response_adbudg=True
        )
        y = model(torch.tensor(
            self.df[[f'v_{k}' for k in range(4, 14)]].values
        ).float())
        assert y.shape[1] == 4

    def test_skipnet_pdim(self):
        model = SkipNet(10, 4, [10,5])
        y = model(torch.tensor(
            self.df[[f'v_{k}' for k in range(4, 14)]].values
        ).float())
        assert y.shape[1] == 4
