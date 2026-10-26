# Fixing SignalLab recognition (real CNN, no fake predictions)

## Why it fails now
This repository includes `backend/artifacts/traffic_sign_cnn.keras` and its matching `model_metadata.json`. The Vercel API installs only Flask/numpy/Pillow; it cannot load TensorFlow or the model. For local inference, install the full backend requirements and run Flask with the project environment. If `/api/health` reports `model_available: false`, verify that the backend is started from this checkout and that the model and metadata files are present.

## Step 1: Train locally
Install Python 3.12. In a terminal at `backend/`:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python train.py --data data/Train --epochs 20 --batch-size 64
```

Download GTSRB from https://benchmark.ini.rub.de/gtsrb_dataset.html and extract 43 numeric folders (00000 through 00042) under `backend/data/Train`. Check the exact expected layout in README.md. Training can take considerable time and RAM. Do not invent accuracy scores.

## Step 2: Verify model
From project root:

```bash
python scripts/check_model.py
```

Both `traffic_sign_cnn.keras` and `model_metadata.json` must exist. Do not edit the model metadata checksum manually. Restart Flask and visit `http://127.0.0.1:5001/api/health`; it must show `model_available: true`. Test with a cropped traffic sign image:

```bash
curl -F 'image=@path/to/sign.png' http://127.0.0.1:5001/api/predict
```

## Step 3: Deploy real inference (separate from Vercel)

The included `backend/Dockerfile` runs the same Flask API using TensorFlow CPU and Gunicorn. Build from the **project root**:

```bash
docker build -f backend/Dockerfile -t signallab-inference .
docker run --rm -p 8080:8080 -e FRONTEND_ORIGIN=https://signallab-traffic-sign-recognition.vercel.app signallab-inference
```

Test `http://localhost:8080/api/health`. A cloud container platform with sufficient memory and persistent model files is needed for deployment. Train locally first, include the trained artifacts in the Docker build context, and keep them private if your dataset/model licensing requires it. A host may have image-size or memory limits. Avoid deploying the large TensorFlow model as a Vercel Python function.

Set `FRONTEND_ORIGIN=https://signallab-traffic-sign-recognition.vercel.app` in the inference service. For Vercel frontend, set `VITE_API_BASE_URL=https://YOUR-ACTUAL-INFERENCE-URL` (without `/api`), then redeploy the frontend. Requests now go to the real inference service, while local development continues using the Vite `/api` proxy. Do not set this variable to the current lightweight Vercel API.

## Troubleshooting

- `MODEL_UNAVAILABLE` / 503: no model, metadata mismatch, or TensorFlow missing. Inspect `/api/health`.
- `Model checksum does not match`: regenerate model and metadata together by training.
- CORS error: set `FRONTEND_ORIGIN` to the exact Vercel origin, without a trailing slash.
- 413: upload a smaller image (browser limit 4 MB; backend request limit 5 MB).
- Cloud memory error: choose a larger instance; TensorFlow can require substantial RAM.
- Wrong predictions: confirm the sign belongs to one of the 43 GTSRB classes. Automatic cropping is a basic aid, not general sign detection; verify with representative images and an independent labelled test set.

**Deployment is not automatic:** a trained model, an accessible inference host and credentials for the owning Vercel project are required.
