"""Run from project root after installing backend/requirements.txt."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"backend"))
from model_loader import ModelStore
s=ModelStore()
print(f"Model status: {s.status}\n{s.message}")
if s.model is None: sys.exit(1)
print("Model output shape:",s.model.output_shape)
