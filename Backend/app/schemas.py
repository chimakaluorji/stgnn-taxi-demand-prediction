from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class UploadResponse(BaseModel):
    inserted_trips: int
    upserted_zones: int
    message: str

class TrainResponse(BaseModel):
    num_zones: int
    num_time_steps: int
    seq_len: int
    epochs: int
    final_train_loss: float
    message: str

class Hotspot(BaseModel):
    zone_id: int
    lat: float
    lng: float
    predicted_demand: float

class PredictResponse(BaseModel):
    forecast_for_timebin_start: datetime
    time_bin_minutes: int
    top_k: int
    hotspots: List[Hotspot]
