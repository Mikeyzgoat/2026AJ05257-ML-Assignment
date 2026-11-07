import pandas as pd
import pytest

from src.app_utils import validate_upload
from src.data_validation import FEATURE_COLUMNS


def test_upload_requires_target():
    with pytest.raises(ValueError):
        validate_upload(pd.DataFrame(columns=FEATURE_COLUMNS))

