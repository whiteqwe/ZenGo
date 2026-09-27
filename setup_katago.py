import os
import urllib.request
import zipfile
import shutil
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KATAGO_DIR = os.path.join(BASE_DIR, 'backend', 'katago')
os.makedirs(KATAGO_DIR, exist_ok=True)

# 1. Download KataGo binary (OpenCL / Eigen)
KATAGO_ZIP_URL = "https://github.com/lightvector/KataGo/releases/download/v1.18.1/katago-v1.18.1-opencl-windows-x64.zip"
KATAGO_EIGEN_URL = "https://github.com/lightvector/KataGo/releases/download/v1.18.1/katago-v1.18.1-eigenavx2-windows-x64.zip"

# 2. 9x9 KataGo Official Neural Network
MODEL_URL = "https://media.katagotraining.org/uploaded/networks/models_extra/kata9x9-b18c384nbt-20231025.bin.gz"

def download_file(url, target_path):
    print(f"Downloading: {url} -> {target_path} ...")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    t0 = time.time()
    with urllib.request.urlopen(req) as resp, open(target_path, 'wb') as f:
        total = int(resp.headers.get('content-length', 0))
        downloaded = 0
        while True:
            chunk = resp.read(1024 * 512)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                percent = downloaded / total * 100
                print(f"\r  Progress: {percent:.1f}% ({downloaded / (1024*1024):.1f} MB / {total / (1024*1024):.1f} MB)", end="")
    print(f"\n  Done in {time.time() - t0:.1f}s")

def setup():
    katago_exe = os.path.join(KATAGO_DIR, 'katago.exe')
    model_path = os.path.join(KATAGO_DIR, 'kata9x9.bin.gz')
    
    # Download KataGo Engine
    if not os.path.exists(katago_exe):
        zip_path = os.path.join(KATAGO_DIR, 'katago.zip')
        try:
            download_file(KATAGO_ZIP_URL, zip_path)
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(KATAGO_DIR)
            if os.path.exists(zip_path):
                os.remove(zip_path)
        except Exception as e:
            print(f"OpenCL download failed: {e}, trying Eigen (CPU)...")
            download_file(KATAGO_EIGEN_URL, zip_path)
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(KATAGO_DIR)
            if os.path.exists(zip_path):
                os.remove(zip_path)

    # Download KataGo 9x9 Model
    if not os.path.exists(model_path):
        download_file(MODEL_URL, model_path)

    print("KataGo setup completed successfully!")

if __name__ == "__main__":
    setup()
