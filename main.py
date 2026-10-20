import os, json, csv
from urllib.parse import urlparse
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE = os.getenv("ICS_BASE_URL", "https://ics.rayanalsubhi.com")
TOKEN = os.getenv("ICS_API_TOKEN", "")
AUTH_STYLE = os.getenv("ICS_AUTH_STYLE", "Token")   # change to "Bearer" if needed
EXPORT_JSON = os.getenv("EXPORT_JSON", "export.json")

if not TOKEN:
    print("Warning: 'ICS_API_TOKEN' environment variable is not set.")
    print("Set it in your environment or in a local .env file.")

OUT_DIR = "partB_dataset"
IMG_DIR = os.path.join(OUT_DIR, "images")
os.makedirs(IMG_DIR, exist_ok=True)

session = requests.Session()
session.headers.update({"Authorization": f"{AUTH_STYLE} {TOKEN}"})

def find_image_ref(task):
    data = task.get("data") or {}
    for k, v in data.items():
        if isinstance(v, str) and ("/data/" in v or v.startswith("http")):
            return v
    return None

def find_text_label(task):
    anns = task.get("annotations") or []
    if not anns:
        return ""
    results = anns[0].get("result") or []
    for r in results:
        val = r.get("value") or {}
        if "text" in val:
            t = val["text"]
            if isinstance(t, list):
                return "".join(t).strip()
            if isinstance(t, str):
                return t.strip()
    return ""

def to_download_url(image_ref: str):
    if image_ref.startswith("http"):
        return image_ref
    parsed = urlparse(image_ref)
    path = parsed.path if parsed.scheme else image_ref
    if path.startswith("/"):
        return BASE + path
    return BASE + "/" + path

with open(EXPORT_JSON, "r", encoding="utf-8") as f:
    tasks = json.load(f)

rows = []
bad = 0

for task in tasks:
    image_ref = find_image_ref(task)
    text = find_text_label(task)

    if not image_ref:
        bad += 1
        continue

    url = to_download_url(image_ref)

    filename = os.path.basename(urlparse(url).path)
    if not filename:
        bad += 1
        continue

    out_path = os.path.join(IMG_DIR, filename)

    if not os.path.exists(out_path):
        r = session.get(url, timeout=90)
        if r.status_code != 200:
            bad += 1
            continue
        with open(out_path, "wb") as wf:
            wf.write(r.content)

    rows.append((os.path.join("images", filename), text))

csv_path = os.path.join(OUT_DIR, "labels.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["image_path", "text"])
    w.writerows(rows)

print("Done")
print("Saved:", csv_path)
print("Images downloaded:", len(rows))
print("Tasks skipped:", bad)
