import json
from pathlib import Path
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from class_names import CLASS_NAMES

def save_json(path, data):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False))
    temp.replace(path)

def evaluate(model, x, y, output, split="validation"):
    scores = model.predict(x, verbose=0)
    predicted = np.argmax(scores, axis=1)
    report = classification_report(y, predicted, labels=list(range(43)), target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    loss, accuracy = model.evaluate(x,y,verbose=0)
    metrics = {"split": split, "samples": len(y), "loss":float(loss), "accuracy":float(accuracy), "report":report}
    save_json(output / "evaluation_metrics.json", metrics)
    save_json(output / "confusion_matrix.json", {"split":split, "labels":CLASS_NAMES, "matrix":confusion_matrix(y,predicted,labels=list(range(43))).tolist()})
    return metrics

if __name__ == "__main__":
    import argparse
    import tensorflow as tf
    from train import load_dataset
    parser=argparse.ArgumentParser(description="Evaluate a separate, labelled test folder (numeric class directories).")
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--output", default="test-results")
    args=parser.parse_args()
    from model_loader import ModelStore
    store=ModelStore()
    if store.model is None:
        parser.error(store.message)
    x,y,_=load_dataset(Path(args.test_data))
    output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    evaluate(store.model,x,y,output,"independent test")
