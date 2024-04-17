import numpy as np
import pandas as pd
from modules.preprocessing.binarizing import (
    DummyPreProcessing, DummyListPreProcessing
)

CATS = ('c_' + pd.Series(np.arange(99).astype(str))).values

TEST_DATA = pd.DataFrame({
    'cat_list': list(map(
        lambda x: np.random.choice(
            CATS, np.random.randint(0, 5, 1)
        ).tolist(),
        range(800)
    )),
    'cat_single': np.random.choice(CATS, 800)
})


class TestDummyPreProcessing:

    def test_transform(self):

        df = DummyPreProcessing(['cat_single']).transform(TEST_DATA)
        assert all(df[list(filter(
            lambda c: 'cat_single_' in c, df.columns
        ))].values.sum(1) == 1)


class TestDummyListPreProcessing:

    def test_transform(self):

        df = DummyListPreProcessing(
            ['cat_list'], {'cat_list': CATS}
        ).transform(TEST_DATA)
        for c in CATS:
            assert f'cat_list_{c}' in df
