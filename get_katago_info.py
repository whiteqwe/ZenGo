import urllib.request
import json
import os
import zipfile
import shutil

url = 'https://api.github.com/repos/lightvector/KataGo/releases/latest'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode('utf-8'))
        print('Latest KataGo release:', data.get('tag_name'))
        for asset in data.get('assets', []):
            if 'windows' in asset['name'].lower():
                print(f"Asset: {asset['name']} -> {asset['browser_download_url']}")
except Exception as e:
    print('Failed to fetch releases:', e)
