import urllib.request
import re

url = "https://katagotraining.org/extra_networks/"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req, timeout=10) as resp:
        html = resp.read().decode('utf-8')
        links = re.findall(r'href=[\'"](https?://[^\'"]+\.bin\.gz)[\'"]', html)
        print("Found models on extra_networks:")
        for l in links:
            print(" ", l)
except Exception as e:
    print("Error:", e)

# Also check main networks page for 6b / 18b / 28b models
url2 = "https://katagotraining.org/networks/"
try:
    with urllib.request.urlopen(urllib.request.Request(url2, headers={'User-Agent': 'Mozilla/5.0'}), timeout=10) as resp:
        html2 = resp.read().decode('utf-8')
        links2 = re.findall(r'href=[\'"](https?://[^\'"]+\.bin\.gz)[\'"]', html2)
        print("Found models on main networks:")
        for l in links2[:5]:
            print(" ", l)
except Exception as e:
    print("Error 2:", e)
