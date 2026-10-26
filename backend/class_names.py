import json
from pathlib import Path
CLASSES = json.loads(Path(__file__).with_name("classes.json").read_text())
CLASS_NAMES = [c["name"] for c in CLASSES]
