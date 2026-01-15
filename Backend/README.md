# ST-GNN Taxi High-Demand Zone Prediction (Backend)

This is a **FastAPI + MongoDB** backend that:

1. Uploads Excel taxi GPS/trip data into MongoDB
2. Preprocesses & aggregates demand per zone/time-bin
3. Constructs a spatial graph from zone centroids
4. Trains a simple **Spatiotemporal Graph Neural Network (ST-GNN)** (GraphConv + GRU)
5. Returns top predicted high-demand zones as (lat,lng) for Google Maps

## 1) Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` (or set env vars):

```bash
export MONGO_URI="mongodb://localhost:27017"
export MONGO_DB="taxi_stgnn"
export MODEL_PATH="models/stgnn_checkpoint.pt"
export TIME_BIN_MINUTES="30"
export SEQ_LEN="12"
export TOP_K="5"
export TRAIN_EPOCHS="15"
export BATCH_SIZE="32"
export LR="1e-3"
export DEVICE="cpu"
```

## 2) Run API

```bash
# Activate virtual environment first
.venv\Scripts\activate  # On Windows
# or
source .venv/bin/activate  # On Linux/Mac

# Then run with virtual environment's Python
python -m uvicorn app.main:app --reload
```

**Windows users can also use the convenience script:**

```bash
run_server.bat
```

Open docs:

- http://127.0.0.1:8000/docs

## 3) Endpoints

### POST /upload_excel

Upload an `.xlsx` with sheets:

- `trips` (trip_id, pickup_datetime, pickup_lat, pickup_lng, pickup_zone_id, ...)
- `zones` (zone_id, zone_centroid_lat, zone_centroid_lng, zone_name)

### POST /process_train

Loads trips/zones from MongoDB, aggregates demand, builds adjacency, trains ST-GNN, saves checkpoint.

### GET /predict_hotspots?top_k=5

Returns JSON with top-K zones and their centroids (lat/lng), plus predicted demand.

## Notes

- Training inside an API is okay for demos, but for production you usually train offline and only serve inference.
- The dummy Excel you generated contains 2 sheets (`trips`, `zones`).
