from typing import Optional
from pymongo import MongoClient
from pymongo.database import Database
from .config import settings

_client: Optional[MongoClient] = None

def get_client() -> MongoClient:
    global _client
    if _client is None:
        _client = MongoClient(settings.MONGO_URI)
    return _client

def get_db() -> Database:
    client = get_client()
    return client[settings.MONGO_DB]

def close_client() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None
