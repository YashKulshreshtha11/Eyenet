from pymongo import MongoClient
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

MONGO_HOST = os.getenv("MONGODB_HOST", "localhost")
MONGO_PORT = int(os.getenv("MONGODB_PORT", 27017))
MONGO_DB = os.getenv("MONGODB_DATABASE_NAME", "eyenet_db")
HIST_COL = os.getenv("MONGODB_HISTORY_COLLECTION", "prediction_history")

print(f"Connecting to {MONGO_HOST}:{MONGO_PORT}...")
try:
    if MONGO_HOST.startswith("mongodb://") or MONGO_HOST.startswith("mongodb+srv://"):
        client = MongoClient(MONGO_HOST, serverSelectionTimeoutMS=5000)
    else:
        client = MongoClient(host=MONGO_HOST, port=MONGO_PORT, serverSelectionTimeoutMS=5000)
    
    db = client[MONGO_DB]
    count = db[HIST_COL].count_documents({})
    print(f"Database: {MONGO_DB}")
    print(f"Collection: {HIST_COL}")
    print(f"Number of documents: {count}")
    
    if count > 0:
        print("First document snippet:")
        print(db[HIST_COL].find_one({}, {"_id": 0, "image_name": 1, "timestamp": 1}))
    
except Exception as e:
    print(f"Error: {e}")
finally:
    client.close()
