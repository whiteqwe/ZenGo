import os
import sys
import uvicorn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)

if __name__ == "__main__":
    print(f"[ZenGo] Starting Go Web Server from: {BASE_DIR}")
    print(f"[ZenGo] Web URL: http://localhost:8000")
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)
