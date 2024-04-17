import pytest
import numpy as np
import pandas as pd

from modules.data_retrieval.aurora_harvest import get_aurora_query
from modules.data_retrieval.df_handlers import select_first, loads_list_str, \
    parse_target_names


class TestSelectFirst:

    df = pd.DataFrame(
        {
            "a": np.concatenate(
                [np.ones((80, 1)), np.random.rand(80, 4)], axis=1
            ).tolist(),
            "b": np.concatenate(
                [np.ones((80, 1)) * 2, np.random.rand(80, 4)], axis=1
            ).tolist(),
            "c": np.concatenate(
                [np.array(["j"] * 80)[:, None], np.random.rand(80, 4)], axis=1
            ).tolist(),
        },
        index=np.arange(80),
    )

    @pytest.mark.dependencies(name="SelectFirst:elements")
    def test_first_elements(self):
        self.df = select_first(self.df, ["a", "c"])
        assert all(self.df["a"]) == 1
        assert self.df["c"].values.astype(str).tolist() == ["j"] * 80

    @pytest.mark.dependencies(
        name="SelectFirst:dtypes", depends=["SelectFirst:elements"]
    )
    def test_dtypes(self):
        assert type(self.df.iloc[0, 2]) == str
        assert type(self.df.iloc[0, 0]) == np.float64


class TestLoadsListStr:

    df = pd.DataFrame(
        {
            "a": [
                "[" + f"{k}," + ",".join([f"{j}" for j in range(10)]) + "]"
                for k in range(10)
            ],
            "b": [
                "[" + f"{k}" + ",".join([f"{j}" for j in range(10)]) + "]"
                for k in range(10)
            ],
            "c": [
                "[" + ",".join(["{" + f'"{j}-1":{j**2}' + "}" for j in range(10)]) + "]"
                for k in range(10)
            ],
        }
    )

    @pytest.mark.dependencies(name="LoadsListStr:elements")
    def test_list_elements(self):
        self.df = loads_list_str(self.df, ["a", "c"])
        assert self.df.shape == (10, 3)
        assert type(self.df["a"][0]) == list
        assert type(self.df["c"][0]) == list
        assert type(self.df["a"][0][0]) in (int, np.int32, np.int64)
        assert type(self.df["c"][0][0]) == dict
        assert len(self.df["a"][0]) == 11
        assert len(self.df["c"][0]) == 10
