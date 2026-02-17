from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging
from rq import Queue
import os
import time

from utils import (
    create_redis_connection, 
    get_redis_config, 
    get_redis_connection,
    clear_model_cache
)
from utils.prediction import predict_house_price

# Set up logging
logger = logging.getLogger(__name__)

app = FastAPI(title="Prediction API", version="1.0.0")
 

# Set up templates directory (for /ui)
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
if os.path.exists(templates_dir):
    templates = Jinja2Templates(directory=templates_dir)
else:
    templates = None

# Redis connection for RQ (lazy initialization)
# Only connects when actually needed, not at import time
redis_conn = None
prediction_queue = None

def init_rq():
    """Initialize RQ (Redis connection + queue) on first use."""
    global redis_conn, prediction_queue
    if redis_conn is None:
        redis_conn = get_redis_connection()
        if redis_conn:
            prediction_queue = Queue("predictions", connection=redis_conn)
            logger.info("Redis connection and queue initialized")
        else:
            logger.warning("Could not connect to Redis. RQ functionality will be limited.")
            prediction_queue = None
    return redis_conn, prediction_queue


class PredictionRequest(BaseModel):
    """Request model matching future_unseen_examples.csv columns."""
    bedrooms: float
    bathrooms: float
    sqft_living: float
    sqft_lot: float
    floors: float
    waterfront: float
    view: float
    condition: float
    grade: float
    sqft_above: float
    sqft_basement: float
    yr_built: float
    yr_renovated: float
    zipcode: int
    lat: float
    long: float
    sqft_living15: float
    sqft_lot15: float


class SimplePredictionRequest(BaseModel):
    """Bonus endpoint request model with only model-required features."""
    bedrooms: float
    bathrooms: float
    sqft_living: float
    sqft_lot: float
    floors: float
    sqft_above: float
    sqft_basement: float
    zipcode: int


class PredictionResponse(BaseModel):
    prediction: float
    confidence: Optional[float] = None


class JobResponse(BaseModel):
    job_id: str
    status: str
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    # Keep this flexible: older clients may expect a float, UI expects an object with `prediction`
    result: Optional[Any] = None
    error: Optional[str] = None


@app.get("/")
async def root():
    return {"message": "Welcome to the Prediction API"}


@app.get("/health")
async def health():
    # Verify Redis connectivity with a simple round-trip.
    # This is intentionally minimal: write a short-lived key and read it back.
    redis_conn = get_redis_connection()
    if redis_conn is None:
        raise HTTPException(status_code=503, detail="Redis not available")

    key = "healthcheck:prediction-api"
    value = f"ok:{time.time()}"
    try:
        redis_conn.set(key, value.encode("utf-8"), ex=30)
        raw = redis_conn.get(key)
        readback = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
        if readback != value:
            raise RuntimeError("Redis round-trip mismatch")
    except Exception as e:
        logger.error(f"Redis healthcheck failed: {e}")
        raise HTTPException(status_code=503, detail="Redis healthcheck failed")

    return {"status": "healthy", "redis_roundtrip": True}


@app.get("/ui", response_class=HTMLResponse)
async def serve_ui(request: Request):
    """Serve the prediction UI page."""
    if templates:
        return templates.TemplateResponse("index.html", {"request": request})

    # Fallback: return a simple message if templates directory doesn't exist
    return HTMLResponse(content="""
    <html>
        <head><title>UI Not Available</title></head>
        <body style="font-family: Arial, sans-serif; padding: 40px; text-align: center;">
            <h1>UI Not Available</h1>
            <p>Please ensure the 'templates' directory exists with index.html</p>
        </body>
    </html>
    """)

@app.post("/predict", response_model=JobResponse)
async def predict(request: PredictionRequest):
    """
    Predict endpoint that enqueues a prediction job and returns immediately.
    """
    _, queue = init_rq()
    if queue is None:
        raise HTTPException(status_code=503, detail="Redis not available")
    
    try:
        features_dict = request.dict()
        
        job = queue.enqueue('worker.predict_house_price_task', features_dict, job_timeout='5m')
        
        return JobResponse(
            job_id=job.id,
            status="queued",
            message=f"Prediction job {job.id} enqueued. Check /predict/job/{job.id}"
        )
    except Exception as e:
        logger.error(f"Error enqueueing job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/simple", response_model=PredictionResponse)
async def predict_simple(request: SimplePredictionRequest):
    """
    Bonus endpoint: Accepts only minimum features.
    """
    try:
        features_dict = request.dict()
        prediction = predict_house_price(features_dict, model_name="simple")
        return PredictionResponse(prediction=prediction, confidence=1.0)
    except Exception as e:
        logger.error(f"Error in simple prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predict/job/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Get the status and result of a prediction job.
    """
    conn, _ = init_rq()
    if conn is None:
        raise HTTPException(status_code=503, detail="Redis not available")
    
    try:
        from rq.job import Job
        job = Job.fetch(job_id, connection=conn)
        status = job.get_status()
        
        result = None
        if status == "finished":
            # UI expects an object with `prediction`; other consumers may handle float/dict.
            jr = job.result
            if isinstance(jr, dict):
                result = jr
            else:
                result = {"prediction": jr}

        return JobStatusResponse(
            job_id=job_id,
            status=status,
            result=result,
            error=str(job.exc_info) if status == "failed" else None
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")


@app.post("/predict/sync", response_model=PredictionResponse)
async def predict_sync(request: PredictionRequest):
    """
    Synchronous predict endpoint.
    """
    try:
        features_dict = request.dict()
        prediction = predict_house_price(features_dict)
        return PredictionResponse(prediction=prediction, confidence=1.0)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/model/register")
async def register_model(
    model_file: UploadFile = File(...),
    model_name: str = Form(...)
):
    """Register a new model with a custom name."""
    redis_conn = get_redis_connection()
    if redis_conn is None:
        raise HTTPException(status_code=503, detail="Redis not available")
    
    content = await model_file.read()
    redis_conn.set(f"model:{model_name}", content)
    
    # Clear cache for this model
    clear_model_cache(model_name)
    
    return {"model_name": model_name, "status": "registered"}


@app.post("/model/set-active")
async def set_active_model(model_name: str):
    """Set the active model to use for predictions."""
    redis_conn = get_redis_connection()
    if redis_conn is None:
        raise HTTPException(status_code=503, detail="Redis not available")
    
    # Verify model exists
    if redis_conn.get(f"model:{model_name}") is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    redis_conn.set("model:active", model_name)
    return {"active_model": model_name, "status": "set"}


@app.get("/model/list")
async def list_models():
    """List all registered models."""
    redis_conn = get_redis_connection()
    if redis_conn is None:
        raise HTTPException(status_code=503, detail="Redis not available")
    
    keys = [k.decode('utf-8').replace('model:', '') for k in redis_conn.keys('model:*') if k != b'model:active']
    active = redis_conn.get("model:active")
    active_name = active.decode('utf-8') if active else None
    return {"models": keys, "active": active_name}
