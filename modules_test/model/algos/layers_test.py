import torch
from modules.model.algos.layers import ADBUDGResponse


def test_adbudg():
    h = torch.rand(20, 7)
    layer = ADBUDGResponse(7)
    g = layer(h)
    assert (g != h).sum() > 0
