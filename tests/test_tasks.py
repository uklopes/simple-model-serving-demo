"""
Minimal unit tests for the background worker task.
"""

from unittest.mock import patch


@patch("worker.predict_house_price")
def test_predict_house_price_task_calls_core_predict(mock_predict, sample_feature_names, sample_features_dict):
    from worker import predict_house_price_task

    mock_predict.return_value = 500000.0
    result = predict_house_price_task(sample_features_dict)

    assert result == 500000.0
    mock_predict.assert_called_once_with(sample_features_dict)


@patch("worker.predict_house_price")
def test_predict_house_price_task_raises_on_error(mock_predict, sample_features_dict):
    from worker import predict_house_price_task

    mock_predict.side_effect = RuntimeError("boom")
    try:
        predict_house_price_task(sample_features_dict)
        assert False, "Expected RuntimeError"
    except RuntimeError:
        pass
