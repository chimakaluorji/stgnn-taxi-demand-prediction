from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, List
import numpy as np
import pandas as pd
from datetime import datetime
import math

def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    # great-circle distance
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlmb/2)**2
    return 2*R*math.asin(math.sqrt(a))

def build_adjacency_from_centroids(zones_df: pd.DataFrame, k: int = 3) -> np.ndarray:
    """Build a weighted adjacency matrix using kNN on centroid haversine distances."""
    zones_df = zones_df.sort_values("zone_id").reset_index(drop=True)
    lats = zones_df["zone_centroid_lat"].to_numpy()
    lngs = zones_df["zone_centroid_lng"].to_numpy()
    n = len(zones_df)
    dist = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                dist[i, j] = 0.0
            else:
                dist[i, j] = _haversine_km(lats[i], lngs[i], lats[j], lngs[j])
    # sigma as median of non-zero distances
    sigma = np.median(dist[dist > 0]) if np.any(dist > 0) else 1.0
    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        nn = np.argsort(dist[i])[1:k+1]  # exclude itself
        for j in nn:
            w = math.exp(-(dist[i, j]**2) / (sigma**2 + 1e-9))
            A[i, j] = w
            A[j, i] = w  # undirected
    return A

@dataclass
class DemandTensor:
    # X: [T, N, 1] normalized
    X: np.ndarray
    # y: [T, N] original demand (not used directly in training; targets created by sliding window)
    demand_matrix: np.ndarray
    time_index: pd.DatetimeIndex
    zone_ids: List[int]
    mean: np.ndarray  # [N]
    std: np.ndarray   # [N]

def aggregate_demand(trips_df: pd.DataFrame, zones_df: pd.DataFrame, time_bin_minutes: int) -> DemandTensor:
    """Aggregate trips into zone-level pickup demand over fixed time bins."""
    trips = trips_df.copy()
    trips["pickup_datetime"] = pd.to_datetime(trips["pickup_datetime"], errors="coerce")
    trips = trips.dropna(subset=["pickup_datetime", "pickup_zone_id"])
    trips["pickup_zone_id"] = trips["pickup_zone_id"].astype(int)

    # bin timestamps
    bin_rule = f"{time_bin_minutes}min"
    trips["time_bin"] = trips["pickup_datetime"].dt.floor(bin_rule)

    # group count
    g = trips.groupby(["time_bin", "pickup_zone_id"]).size().reset_index(name="demand")
    zones_sorted = zones_df.sort_values("zone_id")
    zone_ids = zones_sorted["zone_id"].astype(int).tolist()

    # build full grid (time x zone) fill missing with 0
    time_index = pd.date_range(g["time_bin"].min(), g["time_bin"].max(), freq=bin_rule)
    
    # Create complete index for all time-zone combinations
    full_index = pd.MultiIndex.from_product([time_index, zone_ids], names=['time_bin', 'pickup_zone_id'])
    
    # Reindex to ensure all combinations exist, fill with 0
    g_full = g.set_index(['time_bin', 'pickup_zone_id']).reindex(full_index, fill_value=0).reset_index()
    
    full = (
        g_full.pivot(index="time_bin", columns="pickup_zone_id", values="demand")
         .sort_index()
    )
    demand_matrix = full.to_numpy(dtype=float)  # [T, N]

    # normalize per zone
    mean = demand_matrix.mean(axis=0)
    std = demand_matrix.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)  # Avoid division by zero
    X = (demand_matrix - mean) / std
    X = X[..., None]  # [T, N, 1]
    return DemandTensor(X=X, demand_matrix=demand_matrix, time_index=time_index, zone_ids=zone_ids, mean=mean, std=std)

def make_sliding_windows(X: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create (inputs, targets) for next-step forecasting.
    Inputs: [M, seq_len, N, 1], Targets: [M, N]
    """
    T, N, F = X.shape
    if T <= seq_len:
        raise ValueError(f"Not enough time steps ({T}) for seq_len={seq_len}")
    inputs = []
    targets = []
    for t in range(T - seq_len):
        inputs.append(X[t:t+seq_len])
        # target is next step demand (normalized), shape [N, 1] -> [N]
        targets.append(X[t+seq_len, :, 0])
    return np.stack(inputs, axis=0), np.stack(targets, axis=0)
