from __future__ import annotations
import io
import os
from datetime import datetime, timedelta
from typing import Optional, List

import pandas as pd
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException, Query

from .config import settings
from .db import get_db, close_client
from .schemas import UploadResponse, TrainResponse, PredictResponse, Hotspot
from .pipeline import aggregate_demand, build_adjacency_from_centroids, make_sliding_windows
from .model import train_stgnn, STGNN, predict_next

app = FastAPI(title="ST-GNN Taxi High-Demand Zone Prediction (Backend)")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _ensure_indexes(db):
    db.trips.create_index("pickup_datetime")
    db.trips.create_index("pickup_zone_id")
    db.zones.create_index("zone_id", unique=True)
    db.model_meta.create_index("created_at")
    db.predictions.create_index("created_at")

@app.on_event("startup")
def startup():
    db = get_db()
    _ensure_indexes(db)

@app.on_event("shutdown")
def shutdown():
    close_client()

@app.post("/upload_excel", response_model=UploadResponse)
def upload_excel(file: UploadFile = File(...)):
    """Upload an Excel file (sheets: trips, zones) and store it in MongoDB."""
    if not file.filename.lower().endswith((".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="Please upload an Excel file (.xlsx/.xls).")

    raw = file.file.read()
    try:
        xl = pd.ExcelFile(io.BytesIO(raw))
        trips_df = pd.read_excel(xl, sheet_name="trips")
        zones_df = pd.read_excel(xl, sheet_name="zones")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read Excel: {e}")

    # basic cleaning / typing
    for col in ["pickup_datetime", "dropoff_datetime"]:
        if col in trips_df.columns:
            trips_df[col] = pd.to_datetime(trips_df[col], errors="coerce")
    trips_df = trips_df.dropna(subset=["pickup_datetime", "pickup_lat", "pickup_lng", "pickup_zone_id"])

    # Upsert zones
    db = get_db()
    upserted = 0
    zones_df = zones_df.dropna(subset=["zone_id", "zone_centroid_lat", "zone_centroid_lng"])
    for _, row in zones_df.iterrows():
        z = {
            "zone_id": int(row["zone_id"]),
            "zone_name": str(row.get("zone_name", f"Zone_{int(row['zone_id'])}")),
            "zone_centroid_lat": float(row["zone_centroid_lat"]),
            "zone_centroid_lng": float(row["zone_centroid_lng"]),
            "updated_at": datetime.utcnow(),
        }
        res = db.zones.update_one({"zone_id": z["zone_id"]}, {"$set": z}, upsert=True)
        if res.upserted_id is not None:
            upserted += 1

    # Insert trips (bulk)
    records = trips_df.to_dict(orient="records")
    # make datetimes Python-native (pandas Timestamp -> datetime)
    for r in records:
        r["pickup_datetime"] = pd.to_datetime(r["pickup_datetime"]).to_pydatetime()
        if "dropoff_datetime" in r and not pd.isna(r["dropoff_datetime"]):
            r["dropoff_datetime"] = pd.to_datetime(r["dropoff_datetime"]).to_pydatetime()
        r["pickup_zone_id"] = int(r["pickup_zone_id"])
        if "dropoff_zone_id" in r and not pd.isna(r["dropoff_zone_id"]):
            r["dropoff_zone_id"] = int(r["dropoff_zone_id"])
        r["created_at"] = datetime.utcnow()

    inserted = 0
    if records:
        inserted = len(db.trips.insert_many(records).inserted_ids)

    return UploadResponse(
        inserted_trips=inserted,
        upserted_zones=upserted,
        message="Excel uploaded. Trips stored in MongoDB."
    )

@app.post("/process_train", response_model=TrainResponse)
def process_and_train():
    """Preprocess + aggregate demand, build graph, train ST-GNN, and save model checkpoint."""
    db = get_db()
    trips = list(db.trips.find({}, {"_id": 0}))
    zones = list(db.zones.find({}, {"_id": 0}))

    if not trips or not zones:
        raise HTTPException(status_code=400, detail="MongoDB must contain trips and zones. Upload Excel first.")

    trips_df = pd.DataFrame(trips)
    zones_df = pd.DataFrame(zones)

    # Aggregate demand tensor
    demand = aggregate_demand(trips_df, zones_df, time_bin_minutes=settings.TIME_BIN_MINUTES)

    # Build adjacency (graph)
    A = build_adjacency_from_centroids(zones_df[["zone_id","zone_centroid_lat","zone_centroid_lng"]].copy(), k=3)

    # Sliding windows for training
    X_in, y = make_sliding_windows(demand.X, seq_len=settings.SEQ_LEN)

    # Train model
    artifacts, final_loss = train_stgnn(
        X_in=X_in, y=y, A=A,
        zone_ids=demand.zone_ids, mean=demand.mean, std=demand.std,
        time_bin_minutes=settings.TIME_BIN_MINUTES,
        seq_len=settings.SEQ_LEN,
        epochs=settings.TRAIN_EPOCHS, batch_size=settings.BATCH_SIZE,
        lr=settings.LR, device=settings.DEVICE,
    )

    # Save checkpoint (torch)
    os.makedirs(os.path.dirname(settings.MODEL_PATH), exist_ok=True)
    torch.save({
        "state_dict": artifacts.state_dict,
        "A_norm": artifacts.A_norm,
        "zone_ids": artifacts.zone_ids,
        "mean": artifacts.mean,
        "std": artifacts.std,
        "time_bin_minutes": artifacts.time_bin_minutes,
        "seq_len": artifacts.seq_len,
        "saved_at": datetime.utcnow().isoformat(),
    }, settings.MODEL_PATH)

    # store meta
    db.model_meta.insert_one({
        "created_at": datetime.utcnow(),
        "model_path": settings.MODEL_PATH,
        "final_train_loss": final_loss,
        "num_zones": len(artifacts.zone_ids),
        "num_time_steps": int(demand.X.shape[0]),
        "seq_len": settings.SEQ_LEN,
        "time_bin_minutes": settings.TIME_BIN_MINUTES,
    })

    return TrainResponse(
        num_zones=len(artifacts.zone_ids),
        num_time_steps=int(demand.X.shape[0]),
        seq_len=settings.SEQ_LEN,
        epochs=settings.TRAIN_EPOCHS,
        final_train_loss=final_loss,
        message="Training complete. Model checkpoint saved."
    )

@app.get("/predict_hotspots", response_model=PredictResponse)
def predict_hotspots(top_k: int = Query(default=None, ge=1, le=50)):
    """Return top-K predicted high-demand zones as (lat,lng) for Google Maps."""
    k = top_k if top_k is not None else settings.TOP_K

    if not os.path.exists(settings.MODEL_PATH):
        raise HTTPException(status_code=400, detail="Model not trained. Call /process_train first.")

    ckpt = torch.load(settings.MODEL_PATH, map_location=settings.DEVICE, weights_only=False)
    zone_ids = ckpt["zone_ids"]
    A_norm = torch.tensor(ckpt["A_norm"], dtype=torch.float32, device=settings.DEVICE)

    model = STGNN(A_norm=A_norm, in_features=1).to(settings.DEVICE)
    model.load_state_dict(ckpt["state_dict"])

    mean = np.array(ckpt["mean"], dtype=float)
    std = np.array(ckpt["std"], dtype=float)
    seq_len = int(ckpt["seq_len"])
    time_bin_minutes = int(ckpt["time_bin_minutes"])

    # recompute latest demand sequence from DB to predict next bin
    db = get_db()
    trips = list(db.trips.find({}, {"_id": 0}))
    zones = list(db.zones.find({}, {"_id": 0}))
    trips_df = pd.DataFrame(trips)
    zones_df = pd.DataFrame(zones)

    demand = aggregate_demand(trips_df, zones_df, time_bin_minutes=time_bin_minutes)

    if demand.X.shape[0] <= seq_len:
        raise HTTPException(status_code=400, detail="Not enough demand history to predict. Upload more data.")

    # ensure zone ordering matches checkpoint
    # demand.zone_ids is sorted; zone_ids from checkpoint are also sorted
    if list(demand.zone_ids) != list(zone_ids):
        # Align columns if mismatch
        # (This should not happen if zones are stable.)
        raise HTTPException(status_code=400, detail="Zone ordering mismatch between data and trained model. Retrain model.")

    X_seq = demand.X[-seq_len:][None, ...]  # [1, seq_len, N, 1] normalized using demand's mean/std
    # But checkpoint mean/std may differ; normalize using checkpoint stats for consistency:
    # convert demand_matrix -> normalize via ckpt mean/std
    dm = demand.demand_matrix
    Xn = ((dm - mean) / std)[..., None]
    X_seq = Xn[-seq_len:][None, ...]

    pred = predict_next(model, X_seq=X_seq, mean=mean, std=std, device=settings.DEVICE)  # [N]
    # top-k zones
    idx = np.argsort(-pred)[:k]
    zones_map = zones_df.set_index("zone_id")

    # forecast time = next bin start
    last_bin = demand.time_index[-1].to_pydatetime()
    forecast_start = last_bin + timedelta(minutes=time_bin_minutes)

    hotspots = []
    for i in idx:
        zid = int(zone_ids[i])
        zrow = zones_map.loc[zid]
        hotspots.append(Hotspot(
            zone_id=zid,
            lat=float(zrow["zone_centroid_lat"]),
            lng=float(zrow["zone_centroid_lng"]),
            predicted_demand=float(pred[i]),
        ))

    # store predictions
    db.predictions.insert_one({
        "created_at": datetime.utcnow(),
        "forecast_for_timebin_start": forecast_start,
        "time_bin_minutes": time_bin_minutes,
        "top_k": k,
        "hotspots": [h.dict() for h in hotspots],
    })

    return PredictResponse(
        forecast_for_timebin_start=forecast_start,
        time_bin_minutes=time_bin_minutes,
        top_k=k,
        hotspots=hotspots,
    )
