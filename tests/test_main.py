"""
Minimal unit tests for the FastAPI service.
"""

from unittest.mock import MagicMock, patch


@patch("main.get_redis_connection")
def test_health_endpoint(mock_get_redis_connection, client):
    mock_redis = MagicMock()
    # emulate redis returning bytes
    mock_redis.get.return_value = b"ok:123"
    mock_redis.set.return_value = True
    mock_get_redis_connection.return_value = mock_redis

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["redis_roundtrip"] is True


@patch("main.predict_house_price")
def test_predict_sync_success(mock_predict, client, sample_house_request):
    mock_predict.return_value = 500000.0
    response = client.post("/predict/sync", json=sample_house_request)
    assert response.status_code == 200
    assert response.json()["prediction"] == 500000.0


@patch("main.predict_house_price")
def test_predict_simple_sync_success(mock_predict, client):
    mock_predict.return_value = 123.45
    payload = {
        "bedrooms": 3.0,
        "bathrooms": 2.5,
        "sqft_living": 2000.0,
        "sqft_lot": 5000.0,
        "floors": 1.5,
        "sqft_above": 1500.0,
        "sqft_basement": 500.0,
        "zipcode": 98118,
    }
    response = client.post("/predict/simple", json=payload)
    assert response.status_code == 200
    assert response.json()["prediction"] == 123.45
    mock_predict.assert_called_once()
    assert mock_predict.call_args.kwargs.get("model_name") == "simple"


@patch("main.predict_house_price")
def test_predict_sync_validation_error(mock_predict, client, sample_house_request):
    mock_predict.side_effect = ValueError("Missing required features")
    response = client.post("/predict/sync", json=sample_house_request)
    assert response.status_code == 400


@patch("main.init_rq")
def test_predict_async_enqueues_task_path(mock_init_rq, client, sample_house_request):
    mock_queue = MagicMock()
    mock_job = MagicMock()
    mock_job.id = "test-job-123"
    mock_queue.enqueue.return_value = mock_job
    mock_init_rq.return_value = (MagicMock(), mock_queue)

    response = client.post("/predict", json=sample_house_request)
    assert response.status_code == 200
    assert mock_queue.enqueue.call_args[0][0] == "worker.predict_house_price_task"


@patch("main.init_rq")
def test_predict_async_no_redis(mock_init_rq, client, sample_house_request):
    mock_init_rq.return_value = (None, None)
    response = client.post("/predict", json=sample_house_request)
    assert response.status_code == 503


@patch("main.init_rq")
def test_job_status_finished(mock_init_rq, client):
    mock_init_rq.return_value = (MagicMock(), MagicMock())

    mock_job = MagicMock()
    mock_job.get_status.return_value = "finished"
    mock_job.result = 1.0
    mock_job.exc_info = None

    with patch("rq.job.Job.fetch", return_value=mock_job):
        response = client.get("/predict/job/abc")
    assert response.status_code == 200
    assert response.json()["status"] == "finished"


@patch("main.init_rq")
def test_job_status_not_found(mock_init_rq, client):
    from rq.job import NoSuchJobError

    mock_init_rq.return_value = (MagicMock(), MagicMock())
    with patch("rq.job.Job.fetch", side_effect=NoSuchJobError()):
        response = client.get("/predict/job/missing")
    assert response.status_code == 404
