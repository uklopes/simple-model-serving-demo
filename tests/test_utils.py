"""
Minimal unit tests for utils.
"""

import json
import pathlib
import tempfile
from unittest.mock import patch


@patch("utils.prediction.load_demographics")
def test_join_demographics_merges_when_zipcode_found(mock_load_demographics):
    from utils import join_demographics

    mock_load_demographics.return_value = {98118: {"ppltn_qty": 123.0}}
    out = join_demographics({"zipcode": 98118, "bedrooms": 3.0})

    assert out["bedrooms"] == 3.0
    assert out["ppltn_qty"] == 123.0


def test_join_demographics_returns_input_when_zipcode_not_in_demographics():
    from utils import join_demographics

    with patch("utils.prediction.load_demographics", return_value={}):
        features = {"zipcode": 98118, "bedrooms": 3.0}
        assert join_demographics(features) == features


def test_load_feature_names_list_format():
    from utils import load_feature_names

    features = ["bedrooms", "bathrooms"]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(features, f)
        path = pathlib.Path(f.name)

    with patch("utils.FEATURES_PATH", path):
        load_feature_names.cache_clear()
        assert load_feature_names() == features

    path.unlink()
