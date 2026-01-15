from pydantic import BaseModel
import os

class Settings(BaseModel):
    MONGO_URI: str = os.getenv("MONGO_URI", "")
    MONGO_DB: str = os.getenv("MONGO_DB", "")
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/stgnn_checkpoint.pt")
    TIME_BIN_MINUTES: int = int(os.getenv("TIME_BIN_MINUTES", "30"))
    SEQ_LEN: int = int(os.getenv("SEQ_LEN", "12"))  # 12 x 30min = past 6 hours
    TOP_K: int = int(os.getenv("TOP_K", "5"))
    TRAIN_EPOCHS: int = int(os.getenv("TRAIN_EPOCHS", "15"))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "32"))
    LR: float = float(os.getenv("LR", "1e-3"))
    DEVICE: str = os.getenv("DEVICE", "cpu")

settings = Settings()
