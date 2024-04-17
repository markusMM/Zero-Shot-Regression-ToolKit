import numpy as np
from modules.preprocessing.util import kgauss


class TestKGauss:

    def test_kgauss(self):
        x = np.arange(-3, 4)
        y = kgauss(x)
        y_ = kgauss(x, 1)
        y__ = kgauss(x, sigma=.9)
        s = np.array([
            0.,
            0.17139488, 0.65309954,
            1.,
            0.65309954, 0.17139488,
            0.
        ])
        assert np.allclose(y, s)
        assert ~np.allclose(y, y_)
        assert ~np.allclose(y, y__)
        assert ~np.allclose(y_, y__)
