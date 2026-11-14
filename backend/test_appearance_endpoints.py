import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import init_db
from fastapi.testclient import TestClient
from main import app

print("Testing app with TestClient...")
client = TestClient(app)
print("TestClient started.")

# 1. Root
r = client.get("/")
print("GET / ->", r.status_code, r.json())
assert r.status_code == 200

# 2. Colors endpoint
r = client.get("/appearance-search/colors")
print("GET /appearance-search/colors ->", r.status_code, r.json())
assert r.status_code == 200
assert "orange" in r.json()["colors"]

# 3. Appearance search for white shirt
r = client.post("/appearance-search", json={"query": "white shirt", "confidence_min": 0.3})
print("POST /appearance-search ('white shirt') ->", r.status_code)
data = r.json()
print("  found:", data.get("found"), "total:", data.get("total"))
assert r.status_code == 200
assert data["found"] is True
assert data["total"] > 0
first = data["results"][0]
print("  First match:")
print("    track_id:", first.get("track_id"))
print("    color:", first.get("upper_color"))
print("    conf:", first.get("upper_color_confidence"))
print("    crop_url:", first.get("crop_url"))

# 4. Appearance search for color not present
r = client.post("/appearance-search", json={"query": "orange shirt", "confidence_min": 0.3})
print("POST /appearance-search ('orange shirt') ->", r.status_code)
data = r.json()
print("  found:", data.get("found"), "total:", data.get("total"))
assert r.status_code == 200
assert data["found"] is False

# 5. Crop static file check if crop_url is present
if first.get("crop_url"):
    crop_path = first["crop_url"].split("http://testserver")[-1]
    print(f"GET {crop_path} ->")
    crop_r = client.get(crop_path)
    print("  status code:", crop_r.status_code, "length:", len(crop_r.content))
    assert crop_r.status_code == 200

print("\nALL APPEARANCE ENDPOINT TESTS PASSED SUCCESSFULLY!")
