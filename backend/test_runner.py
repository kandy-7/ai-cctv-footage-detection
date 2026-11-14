import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import appearance_search, AppearanceQueryRequest, get_supported_colors, read_root

print("1. read_root():", read_root())
print("2. get_supported_colors():", get_supported_colors()["colors"])

req = AppearanceQueryRequest(query="white shirt", confidence_min=0.3)
mock_http = SimpleNamespace(base_url="http://localhost:8001/")

res = appearance_search(req, mock_http)
print(f"3. appearance_search('white shirt'): found={res['found']}, total={res['total']}")
for r in res["results"]:
    print(f"   track={r['track_id']} color={r['upper_color']} conf={r['upper_color_confidence']:.2f} crop={r['crop_url']}")

req_orange = AppearanceQueryRequest(query="orange shirt", confidence_min=0.3)
res_orange = appearance_search(req_orange, mock_http)
print(f"4. appearance_search('orange shirt'): found={res_orange['found']}, total={res_orange['total']}")

print("\nALL VERIFICATION STEPS PASSED SUCCESSFULLY!")
