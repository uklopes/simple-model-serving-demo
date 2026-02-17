---
marp: true
title: Simple Model Serving Demo — Technical Decisions
description: Technical deep dive for the Simple Model Service Demo
paginate: true
theme: default
---

# Simple Model Serving Demo  
## Technical decisions & implementation

Objective: model with a simple UI, scalable execution, and deployable ops.

---

## Architecture (high level)

- **Web UI**: `GET /ui` (Jinja template) for interactive demo + stakeholder-friendly validation
- **FastAPI service**: serves sync and async prediction endpoints
- **Redis**:
  - RQ job queue (async predictions)
  - model registry (model bytes + active model pointer)
- **RQ worker**: executes background prediction jobs

---

## Inference endpoints (contract & UX)

- **Sync (full payload)**: `POST /predict/sync`
  - immediate response: `{"prediction": <float>}`
- **Sync (minimal payload)**: `POST /predict/simple`
  - uses a separately-trained simplified model
- **Async (scalable)**: `POST /predict` → returns `job_id`
  - poll: `GET /predict/job/{job_id}`

---

## Async flow (why RQ)

**Goal:** keep the web service responsive while work scales horizontally.

1) Web receives request → enqueues job to Redis (`predictions` queue)
2) Worker consumes job → runs shared core predictor → stores result in job record
3) Client/UI polls job status endpoint until finished

Scaling: add worker replicas without changing the web layer.

---

## Model lifecycle: registry + “active model”

- **Register**: `POST /model/register` uploads a model artifact into Redis (`model:{name}`)
- **Activate**: `POST /model/set-active?model_name=...` sets `model:active`
- **List**: `GET /model/list`

Design intent:
- model updates without service downtime (swap active pointer)
- supports multiple variants (e.g., `default`, `simple`)

---

## “Simple” model endpoint 

Problem: the full model expects many fields; early-stage scenarios may only have a subset.

Solution:
- train a **separate simplified model** with a reduced input schema
- expose `POST /predict/simple` that only requires those fields


---

## Model training & selection

- **Training script**: `scripts/create_model.py`
- **Features**: used the full available numeric property features and joined ZIP-based demographics
- **Algorithm selection**:
  - tested **Random Forest** and **SVR**
  - moved to **LightGBM** for strong performance compactness
- **Two artifacts produced**:
  - **Complete model** (full input schema)
  - **Simple model** (reduced required inputs)
- **Metrics reported**: the script prints test-set metrics (R², MAE, RMSE, MAPE) for both models and writes artifacts under `model/`

---

## Model metrics (test dataset)

From running `scripts/create_model.py`:

- **Full model**
  - R²: **0.8906**
  - MAE: **$68,418**
  - RMSE: **$127,903**
  - MAPE: **12.80%**
- **Simplified model**
  - R²: **0.8038**
  - MAE: **$90,689**
  - RMSE: **$171,325**
  - MAPE: **16.42%**

---

## Deployment

- **Docker + docker-compose** for local full-stack (web + worker + Redis)
- **Health check**: `GET /health`
- **Railway**: Deployed at https://simple-model-serving-production.up.railway.app

---

## Future

- More Zipcode data
- **Reliability**: startup seeding + readiness checks so workers don’t process jobs before models are available
- **Observability**: metrics and tracing for latency, queue depth, error rates; structured logs with request/job IDs
- **Model governance**: richer model metadata (training time, dataset version, schema hash, metrics), safer rollouts (canary/gradual)
- **Performance**: batch prediction endpoint, caching of feature lists/demographics, reduce per-request overhead