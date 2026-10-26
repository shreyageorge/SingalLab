"""Vercel WSGI entrypoint; reuse the same Flask APIs as local development."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
import app as service

app = service.app
if service.store.model is None:
    service.store.message = (
        "No trained model is deployed. Train locally, then deploy model artifacts "
        "with a TensorFlow-capable inference runtime. See the deployment guide."
    )
