import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

print(f"MONGODB_HOST: {os.getenv('MONGODB_HOST')!r}")
print(f"MONGODB_PORT: {os.getenv('MONGODB_PORT')!r}")
print(f"MONGODB_DATABASE_NAME: {os.getenv('MONGODB_DATABASE_NAME')!r}")
