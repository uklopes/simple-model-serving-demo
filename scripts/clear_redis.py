#!/usr/bin/env python3
import sys
import pathlib

# Add parent directory to path to import local utils
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from utils import get_redis_connection

redis_conn = get_redis_connection()
if not redis_conn:
    print("Error: Could not connect to Redis")
    sys.exit(1)

# Clear all model-related keys
keys = redis_conn.keys('model:*')
if keys:
    redis_conn.delete(*keys)
    print(f"Cleared {len(keys)} model keys")
else:
    print("No model keys found")
