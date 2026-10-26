import json
import io
import os
import time
from pathlib import Path
import numpy as np
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from werkzeug.exceptions import HTTPException
from class_names import CLASSES
from model_loader import ModelStore, ARTIFACTS
from predict import decode_image, inference_array
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024
CORS(app, origins=os.getenv("FRONTEND_ORIGIN", "http://localhost:5173").split(","))
store = ModelStore()
def artifact(name):
    path = ARTIFACTS / name
    if not path.exists():
        return {"available": False, "message": "No training artifacts yet. Configure the dataset and run train.py."}
    return {"available": True, "data": json.loads(path.read_text())}
@app.get("/api/health")
def health():
    return jsonify(status="ok", model_available=store.model is not None, model_status=store.status, message=store.message)
@app.get("/api/classes")
def classes():
    return jsonify(classes=CLASSES)
@app.post("/api/predict")
def predict():
    upload = request.files.get("image")
    if upload is None:
        return jsonify(error="Upload an image using the 'image' form field."), 400
    try:
        image = decode_image(upload.read())
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    if store.model is None:
        return jsonify(error=store.message, code="MODEL_UNAVAILABLE"), 503
    if not store.lock.acquire(blocking=False):
        return jsonify(error="Model is busy. Please retry shortly."), 429
    try:
        start = time.perf_counter()
        scores = np.asarray(store.model(inference_array(image)[None], training=False))[0]
        if scores.shape != (43,) or not np.isfinite(scores).all() or (scores < 0).any() or not np.isclose(scores.sum(),1,atol=1e-3):
            raise ValueError("Model returned invalid softmax scores.")
        elapsed = (time.perf_counter() - start) * 1000
        top = [{"class_id": int(i), "class_name": CLASSES[int(i)]["name"], "score": float(scores[i])} for i in np.argsort(scores)[::-1][:5]]
        return jsonify(**top[0], top_k=top, inference_ms=round(elapsed,2), score_type="softmax")
    finally:
        store.lock.release()
@app.post("/api/preview")
def preview():
    upload=request.files.get("image")
    if upload is None:
        return jsonify(error="Upload an image using the 'image' form field."),400
    try:
        image=decode_image(upload.read())
        image.thumbnail((1024,1024))
        output=io.BytesIO(); image.save(output,format="PNG"); output.seek(0)
        return send_file(output,mimetype="image/png")
    except ValueError as exc:
        return jsonify(error=str(exc)),400
@app.get("/api/metrics")
def metrics():
    return jsonify(history=artifact("training_history.json"), evaluation=artifact("evaluation_metrics.json"), dataset=artifact("dataset_metadata.json"))
@app.get("/api/confusion-matrix")
def matrix():
    return jsonify(artifact("confusion_matrix.json"))
@app.get("/api/model-info")
def model_info():
    return jsonify(store.info())
@app.post("/api/train")
def train():
    return jsonify(error="Web training is disabled. Use the validated CLI command shown in Training to avoid competing jobs."), 405
@app.get("/api/train/status")
def training_status():
    return jsonify(mode="cli", **artifact("training_status.json"))
@app.get("/api/samples")
def samples():
    items=[]
    for c in CLASSES:
        path=ARTIFACTS / "samples" / f"{c['id']}.png"
        if path.exists():
            items.append({"class_id":c["id"], "name":c["name"], "url":f"/api/samples/{c['id']}"})
    return jsonify(samples=items)
@app.get("/api/samples/<int:class_id>")
def sample(class_id):
    path=ARTIFACTS / "samples" / f"{class_id}.png"
    if class_id not in range(43) or not path.exists():
        return jsonify(error="Dataset example unavailable."),404
    return send_file(path, mimetype="image/png")
@app.errorhandler(Exception)
def error(exc):
    if isinstance(exc, HTTPException):
        return jsonify(error=exc.description), exc.code
    app.logger.exception("Request failed")
    return jsonify(error="The backend could not complete this request. Check the server log."),500
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False)
