"""
MongoDB database integration for EyeNet.
Handles prediction history and system logs.
"""

import datetime
import logging
import os
from typing import Any, Dict, List, Optional

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import ConnectionFailure
from bson import ObjectId

logger = logging.getLogger("eyenet.database")


# ─────────────────────────────────────────────────────────────────────────────
# Config (from environment with fallbacks)
# ─────────────────────────────────────────────────────────────────────────────
MONGO_HOST  = os.getenv("MONGODB_HOST", "localhost")
MONGO_PORT  = int(os.getenv("MONGODB_PORT", 27017))
MONGO_DB    = os.getenv("MONGODB_DATABASE_NAME", "eyenet_db")
HIST_COL    = os.getenv("MONGODB_HISTORY_COLLECTION", "prediction_history")
LOGS_COL    = os.getenv("MONGODB_LOGS_COLLECTION", "system_logs")


# ─────────────────────────────────────────────────────────────────────────────
# Connection management
# ─────────────────────────────────────────────────────────────────────────────
_client: Optional[MongoClient] = None
_db: Optional[Database] = None


def get_db() -> Optional[Database]:
    """Return the cached MongoDB database handle (or None if unavailable)."""
    global _client, _db

    if _db is not None:
        return _db

    try:
        _client = MongoClient(
            host=MONGO_HOST,
            port=MONGO_PORT,
            serverSelectionTimeoutMS=3000,
        )
        _client.admin.command("ping")   # fast liveness check
        _db = _client[MONGO_DB]
        logger.info("Connected to MongoDB at %s:%s", MONGO_HOST, MONGO_PORT)
    except ConnectionFailure as exc:
        logger.warning("MongoDB unavailable (%s). History/logs disabled.", exc)
        _db = None

    return _db


def close_db():
    """Close the MongoDB connection."""
    global _client, _db
    if _client is not None:
        _client.close()
        _client = None
        _db = None


# ─────────────────────────────────────────────────────────────────────────────
# Prediction history
# ─────────────────────────────────────────────────────────────────────────────

def save_prediction(
    image_name: str,
    predicted_class: str,
    probabilities: Dict[str, float],
    inference_time_ms: float,
    gradcam_base64: str | None = None,
    overlay_base64: str | None = None,
) -> Optional[str]:
    """
    Persist a prediction record.

    Returns the inserted document id as a string, or None on failure.
    """
    db = get_db()
    if db is None:
        return None

    doc = {
        "image_name":        image_name,
        "timestamp":         datetime.datetime.utcnow(),
        "predicted_class":   predicted_class,
        "probabilities":     probabilities,
        "inference_time_ms": inference_time_ms,
        "gradcam_base64":    gradcam_base64,
        "overlay_base64":    overlay_base64,
    }
    try:
        result = db[HIST_COL].insert_one(doc)
        return str(result.inserted_id)
    except Exception as exc:
        logger.error("Failed to save prediction: %s", exc)
        return None


def get_history(limit: int = 50, skip: int = 0) -> List[Dict]:
    """Return a slice of prediction records (sorted by most recent)."""
    db = get_db()
    if db is None:
        return []

    cursor = (
        db[HIST_COL]
        .find({})
        .sort("timestamp", -1)
        .skip(skip)
        .limit(limit)
    )
    history = []
    for doc in cursor:
        doc["_id"] = str(doc["_id"])
        history.append(doc)
    return history


def get_record_by_id(record_id: str) -> Optional[Dict]:
    """Retrieve a single prediction record by its string ID."""
    db = get_db()
    if db is None:
        return None
    try:
        return db[HIST_COL].find_one({"_id": ObjectId(record_id)})
    except Exception:
        return None


def delete_record(record_id: str) -> bool:
    """Delete a prediction record by its ID. Returns True if successful."""
    db = get_db()
    if db is None:
        return False
    try:
        res = db[HIST_COL].delete_one({"_id": ObjectId(record_id)})
        return res.deleted_count > 0
    except Exception as exc:
        logger.error("Failed to delete record %s: %s", record_id, exc)
        return False


def delete_all_records() -> bool:
    """Delete all prediction records. Returns True if successful."""
    db = get_db()
    if db is None:
        return False
    try:
        db[HIST_COL].delete_many({})
        return True
    except Exception as exc:
        logger.error("Failed to clear history: %s", exc)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# System logs
# ─────────────────────────────────────────────────────────────────────────────

def log_event(
    event_type: str,
    message: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Append a structured log entry (best-effort; silently ignores errors)."""
    db = get_db()
    if db is None:
        return

    doc = {
        "event_type": event_type,
        "message":    message,
        "timestamp":  datetime.datetime.utcnow(),
        "metadata":   metadata or {},
    }
    try:
        db[LOGS_COL].insert_one(doc)
    except Exception as exc:
        logger.error("Failed to write log: %s", exc)
