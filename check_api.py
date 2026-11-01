import json, sys, urllib.request

base = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://localhost:8000"
example = {"discharge_time": 6026.76, "decrement_v": 1220.00, "max_v_discharge": 4.25, "min_v_charge": 3.21,
           "time_415v": 5505.99, "time_cc": 6161.00, "charging_time": 10500.33}

def post(path, body):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=60))

assert json.load(urllib.request.urlopen(base + "/health", timeout=60))["model_loaded"]
rul = post("/predict", example)["predicted_rul"]
assert 1100 < rul < 1117, f"predicted_rul={rul}; ~1108.6 expected (unscaled input gives ~955)"
batch = post("/predict/batch", {"batch": [example, example]})["predictions"]
assert batch == [rul, rul], batch
print(f"OK predicted_rul={rul:.1f}")
