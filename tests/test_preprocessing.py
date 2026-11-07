import pandas as pd

from src.preprocessing import map_target


def test_target_mapping():
    assert map_target(pd.Series([-1, 1, -1])).tolist() == [1, 0, 1]

