import os, json, re, requests, numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# --------- CONFIG ----------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY env var before running.")
HEADERS = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

# Load memory (make sure embeddings.json is in the same folder)
MEM = json.load(open("embeddings.json", "r", encoding="utf-8"))
MEM_MAT = np.array([m["embedding"] for m in MEM], dtype=np.float32)
MEM_MAT = MEM_MAT / (np.linalg.norm(MEM_MAT, axis=1, keepdims=True) + 1e-12)

# High-fashion runway ranges (in cm)
FEMALE_RANGES = {
    "height_cm": (175, 180),
    "bust":  (78, 84),
    "waist": (58, 63),
    "hips":  (85, 90),
}
MALE_RANGES = {
    "height_cm": (183, 190),
    "chest": (90, 100),
    "waist": (73, 82),
    "hips":  (88, 96),
}

# ---- measurement helpers ----
def _to_cm(values, unit="cm"):
    nums = [float(v) for v in values]
    if unit.lower() in ["in", "inch", "inches"]:
        return [round(v * 2.54, 1) for v in nums]
    return nums

def parse_measurements(meas_str):
    if not meas_str or not isinstance(meas_str, str):
        return None
    s = meas_str.strip().lower()
    unit = "cm"
    if " in" in s or s.endswith("in"):
        unit = "in"
    s_clean = re.sub(r"[^0-9\-/ ,.]", " ", s)
    parts = re.split(r"[-/ ,]+", s_clean)
    parts = [p for p in parts if p]
    if len(parts) < 3:
        return None
    vals = parts[:3]
    try:
        nums_cm = _to_cm(vals, unit=unit)
        return {"bust_or_chest": nums_cm[0], "waist": nums_cm[1], "hips": nums_cm[2]}
    except Exception:
        return None

def within(value, lo, hi):
    return value is not None and lo <= value <= hi

# ---- OpenAI helpers ----
def describe(url: str) -> str:
    if not (isinstance(url, str) and url.startswith("https://")):
        raise ValueError(f"Photo URL must be https: {url}")
    body = {
        "model": "gpt-4o-mini",
        "temperature": 0,
        "messages": [
            {"role":"system","content":"Describe the subject for fashion modeling suitability. Max 2 lines. Avoid sensitive attributes; focus on symmetry, bone structure, proportions, skin clarity, editorial/commercial vibe."},
            {"role":"user","content":[
                {"type":"text","text":"Describe this applicant image."},
                {"type":"image_url","image_url":{"url":url}}
            ]}
        ]
    }
    r = requests.post("https://api.openai.com/v1/chat/completions", headers=HEADERS, json=body, timeout=90)
    if r.status_code != 200:
        raise RuntimeError(f"OpenAI error {r.status_code}: {r.text}")
    return r.json()["choices"][0]["message"]["content"].strip()

def embed(txt: str) -> np.ndarray:
    body = {"model":"text-embedding-3-large","input":txt}
    r = requests.post("https://api.openai.com/v1/embeddings", headers=HEADERS, json=body, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"OpenAI error {r.status_code}: {r.text}")
    v = np.array(r.json()["data"][0]["embedding"], dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-12)

app = Flask(__name__)
CORS(app)  # allow Bubble to call

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"ok": True})

@app.route("/evaluate", methods=["POST"])
def evaluate():
    data = request.get_json(force=True) or {}
    photos = data.get("photos", [])[:5]
    gender = data.get("gender")
    height_cm = data.get("height_cm")
    age = data.get("age")
    measurements_str = data.get("measurements")

    if not photos or not isinstance(photos, list):
        return jsonify({"error":"Missing photos[]"}), 400
    if gender not in ["Male", "Female"]:
        return jsonify({"error":"gender must be 'Male' or 'Female'"}), 400

    # Hard rules: age
    if age is not None and (age < 16 or age > 23):
        return jsonify({"decision":"REJECTED","reason":"Age outside 16–23","confidence":0.0,"details":[]})

    # Hard rules: height
    if gender == "Male":
        lo, hi = MALE_RANGES["height_cm"]
        if (height_cm is not None) and not (lo <= height_cm <= hi):
            return jsonify({"decision":"REJECTED","reason":"Male height outside 183–190","confidence":0.0,"details":[]})
    else:
        lo, hi = FEMALE_RANGES["height_cm"]
        if (height_cm is not None) and not (lo <= height_cm <= hi):
            return jsonify({"decision":"REJECTED","reason":"Female height outside 175–180","confidence":0.0,"details":[]})

    details, sims = [], []
    for u in photos:
        try:
            desc = describe(u)
            v = embed(desc)
            s = float(np.max(MEM_MAT @ v))
            sims.append(s)
            details.append({"url": u, "best_similarity": round(s,3), "desc": desc})
        except Exception as e:
            details.append({"url": u, "error": str(e)})

    if not sims:
        return jsonify({"decision":"NEEDS_REVIEW","reason":"No valid photos processed","confidence":0.0,"details":details})

    best = max(sims)

    # Measurements
    parsed = parse_measurements(measurements_str) if measurements_str else None
    if parsed:
        if gender == "Male":
            chest_ok = within(parsed["bust_or_chest"], *MALE_RANGES["chest"])
            waist_ok = within(parsed["waist"], *MALE_RANGES["waist"])
            hips_ok  = within(parsed["hips"],  *MALE_RANGES["hips"])
            if not (chest_ok and waist_ok and hips_ok):
                return jsonify({"decision":"REJECTED","reason":"Male measurements outside runway range",
                                "confidence": round(best,3), "details": details, "parsed_measurements_cm": parsed})
        else:
            bust_ok  = within(parsed["bust_or_chest"], *FEMALE_RANGES["bust"])
            waist_ok = within(parsed["waist"], *FEMALE_RANGES["waist"])
            hips_ok  = within(parsed["hips"],  *FEMALE_RANGES["hips"])
            if not (bust_ok and waist_ok and hips_ok):
                return jsonify({"decision":"REJECTED","reason":"Female measurements outside runway range",
                                "confidence": round(best,3), "details": details, "parsed_measurements_cm": parsed})

    # Similarity thresholds
    if best >= 0.78:
        decision = "SELECTED"
    elif best >= 0.60:
        decision = "NEEDS_REVIEW"
    else:
        decision = "REJECTED"

    return jsonify({
        "decision": decision,
        "confidence": round(best,3),
        "details": details,
        "parsed_measurements_cm": parsed
    })
@app.route("/routes", methods=["GET"])
def routes():
    from flask import jsonify
    rule_map = []
    for r in app.url_map.iter_rules():
        rule_map.append({"endpoint": r.endpoint, "rule": str(r), "methods": sorted(list(r.methods))})
    return jsonify(rule_map)

if __name__ == "__main__":
    # On Render/Railway, they set PORT for you
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)

