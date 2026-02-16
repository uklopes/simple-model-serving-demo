#!/usr/bin/env python3
"""
RQ Worker entry point and background tasks.
"""
import os
import sys
import logging
from typing import Dict, Any
from rq import Worker, Queue, Connection
from redis import Redis

from utils import create_redis_connection, get_redis_config
from utils.prediction import predict_house_price

# Set up logging
logger = logging.getLogger(__name__)


def predict_house_price_task(features: Dict[str, Any]) -> float:
    """
    Background task to process a prediction job.
    This function will be called by RQ worker.
    """
    logger.info(f"Starting background prediction task for zipcode: {features.get('zipcode')}")
    
    try:
        # Core prediction logic is shared between sync and async endpoints
        prediction = predict_house_price(features)
        logger.info(f"Prediction task completed successfully: {prediction}")
        return prediction
    except Exception as e:
        logger.error(f"Error in background task: {e}", exc_info=True)
        raise


queue_name = os.getenv("QUEUE_NAME", "predictions")

if __name__ == "__main__":
    # Connect to Redis
    try:
        redis_conn = create_redis_connection(raise_on_error=True)
        config = get_redis_config()
        print(f"Connected to Redis at {config['host']}:{config['port']}")
    except Exception as e:
        print(f"Error connecting to Redis: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Create queue
    queue = Queue(queue_name, connection=redis_conn)
    
    # Start worker
    print(f"Starting RQ worker for queue: {queue_name}")
    with Connection(redis_conn):
        worker = Worker([queue])
        worker.work()
