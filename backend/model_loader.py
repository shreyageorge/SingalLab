import hashlib
import json
import threading
from pathlib import Path
import numpy as np
from class_names import CLASS_NAMES
ARTIFACTS = Path(__file__).parent / "artifacts"
class ModelStore:
    def __init__(self):
        self.model = None
        self.status = "missing"
        self.message = "Train the baseline CNN, then restart the backend. See README.md."
        self.lock = threading.Lock()
        path = ARTIFACTS / "traffic_sign_cnn.keras"
        if not path.exists():
            return
        try:
            import tensorflow as tf
            metadata = json.loads((ARTIFACTS / "model_metadata.json").read_text())
            if metadata["class_names"] != CLASS_NAMES or metadata["preprocessing"] != "embedded_rescaling_1_255":
                raise ValueError("Class mapping or preprocessing metadata is incompatible.")
            if metadata["model_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError("Model checksum does not match training metadata.")
            model = tf.keras.models.load_model(path, compile=False)
            if tuple(model.input_shape[1:]) != (32,32,3) or tuple(model.output_shape[1:]) != (43,):
                raise ValueError("Expected input (32,32,3) and output (43,).")
            scales = [l for l in model.layers if isinstance(l, tf.keras.layers.Rescaling)]
            if len(scales) != 1 or not np.isclose(scales[0].scale, 1/255) or scales[0].offset != 0:
                raise ValueError("Expected exactly one embedded 1/255 rescaling layer.")
            if model.layers[-1].get_config().get("activation") != "softmax":
                raise ValueError("Output must use softmax.")
            self.model = model
            self.status, self.message = "ready", "Trained baseline model loaded."
        except Exception as exc:
            self.status, self.message = "error", f"Model loading failed: {exc}"
    def info(self):
        if self.model is None:
            return {"available": False, "status": self.status, "message": self.message}
        m = self.model
        return {"available": True, "input_shape": list(m.input_shape), "output_shape": list(m.output_shape),
                "parameters": m.count_params(), "trainable_parameters": sum(int(np.prod(w.shape)) for w in m.trainable_weights),
                "non_trainable_parameters": sum(int(np.prod(w.shape)) for w in m.non_trainable_weights),
                "layers": [{"name": l.name, "type": type(l).__name__, "output_shape": list(l.output.shape), "parameters": l.count_params()} for l in m.layers]}
