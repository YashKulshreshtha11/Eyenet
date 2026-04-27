import requests
import json

resp = requests.get("http://localhost:8001/api/v1/history?limit=1")
print(json.dumps(resp.json(), indent=2))
