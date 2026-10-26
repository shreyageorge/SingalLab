import argparse
import hashlib
import os
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from class_names import CLASS_NAMES
from predict import decode_image, image_array
from evaluate import evaluate, save_json
from model_loader import ARTIFACTS

def track_group(class_id, stem):
    parts = stem.split("_")
    track = "_".join(parts[:2]) if len(parts) > 1 else stem
    return f"{class_id}/{track}"

def split_dataset(y, groups, validation_split):
    train_indices, val_indices = [], []
    for class_id in range(43):
        class_indices = np.flatnonzero(y == class_id)
        class_groups = groups[class_indices]
        if len(np.unique(class_groups)) < 2:
            raise ValueError(f"Class {class_id} needs at least two independent sign tracks.")
        splitter = GroupShuffleSplit(n_splits=1, test_size=validation_split, random_state=42 + class_id)
        train_local, val_local = next(splitter.split(class_indices, y[class_indices], class_groups))
        train_indices.extend(class_indices[train_local])
        val_indices.extend(class_indices[val_local])
    return np.asarray(train_indices), np.asarray(val_indices)

def load_dataset(folder):
    images, labels, groups = [], [], []
    paths=[]
    for class_id in range(43):
        directories = [p for p in folder.iterdir() if p.is_dir() and p.name.isdigit() and int(p.name)==class_id]
        if len(directories)!=1:
            raise ValueError(f"Expected one numeric folder for class {class_id} in {folder}.")
        files=sorted(p for p in directories[0].iterdir() if p.suffix.lower() in {".ppm",".png",".jpg",".jpeg",".bmp",".webp"})
        if not files:
            raise ValueError(f"Class {class_id} has no images.")
        for p in files:
            images.append(image_array(decode_image(p.read_bytes())))
            labels.append(class_id)
            groups.append(track_group(class_id, p.stem))
            paths.append(p)
    return np.asarray(images), np.asarray(labels), (np.asarray(groups),paths)

def build_model(learning_rate):
    import tensorflow as tf
    layers=tf.keras.layers
    model=tf.keras.Sequential([layers.Input((32,32,3)),layers.Rescaling(1/255,name="normalize"),
        layers.Conv2D(32,3,activation="relu",padding="same"), layers.MaxPooling2D(),
        layers.Conv2D(64,3,activation="relu",padding="same"), layers.MaxPooling2D(),
        layers.Conv2D(128,3,activation="relu",padding="same"), layers.MaxPooling2D(),
        layers.Flatten(),layers.Dense(128,activation="relu"),layers.Dropout(.5),layers.Dense(43,activation="softmax")])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    return model

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--epochs",type=int,default=20)
    parser.add_argument("--batch-size",type=int,default=64)
    parser.add_argument("--validation-split",type=float,default=.2)
    parser.add_argument("--learning-rate",type=float,default=.001)
    args=parser.parse_args()
    if not 1<=args.epochs<=200 or not 8<=args.batch_size<=512 or not .1<=args.validation_split<=.4 or not 1e-6<=args.learning_rate<=.01:
        parser.error("epochs 1–200; batch-size 8–512; validation-split .1–.4; learning-rate .000001–.01")
    ARTIFACTS.mkdir(exist_ok=True)
    lock=ARTIFACTS / "training.lock"
    try:
        fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:
        parser.error("Another training job holds training.lock. Remove only after confirming it has stopped.")
    os.close(fd)
    try:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(42)
        tf.config.experimental.enable_op_determinism()
        x,y,(groups,paths)=load_dataset(args.data)
        train_idx,val_idx=split_dataset(y,groups,args.validation_split)
        model=build_model(args.learning_rate)
        checkpoint=ARTIFACTS / "best_checkpoint.keras"
        class Progress(tf.keras.callbacks.Callback):
            def on_epoch_end(self,epoch,logs=None):
                save_json(ARTIFACTS/"training_status.json",{"status":"running","epoch":epoch+1,"epochs":args.epochs,"metrics":{k:float(v) for k,v in (logs or {}).items()}})
        save_json(ARTIFACTS/"training_status.json",{"status":"running","epoch":0,"epochs":args.epochs})
        history=model.fit(x[train_idx],y[train_idx],validation_data=(x[val_idx],y[val_idx]),epochs=args.epochs,batch_size=args.batch_size,
            callbacks=[Progress(),tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=5,restore_best_weights=True),tf.keras.callbacks.ModelCheckpoint(checkpoint,monitor="val_loss",save_best_only=True)])
        model=tf.keras.models.load_model(checkpoint)
        target=ARTIFACTS/"traffic_sign_cnn.keras"
        temp=ARTIFACTS/"new_model.keras"; model.save(temp); temp.replace(target)
        save_json(ARTIFACTS/"model_metadata.json",{"class_names":CLASS_NAMES,"preprocessing":"embedded_rescaling_1_255","model_sha256":hashlib.sha256(target.read_bytes()).hexdigest(),"seed":42})
        h={k:[float(v) for v in values] for k,values in history.history.items()}
        h["best_epoch"]=int(np.argmin(h["val_loss"]))+1
        save_json(ARTIFACTS/"training_history.json",h)
        save_json(ARTIFACTS/"dataset_metadata.json",{"training_samples":len(train_idx),"validation_samples":len(val_idx),"classes":43,"split_method":"Group split by sign track (seed 42)","class_counts":np.bincount(y,minlength=43).tolist()})
        evaluate(model,x[val_idx],y[val_idx],ARTIFACTS)
        samples=ARTIFACTS/"samples"; samples.mkdir(exist_ok=True)
        for i in range(43):
            p=paths[next(j for j in val_idx if y[j]==i)]
            decode_image(p.read_bytes()).save(samples/f"{i}.png")
        save_json(ARTIFACTS/"training_status.json",{"status":"complete","epochs_completed":len(h["loss"]),"best_epoch":h["best_epoch"]})
    except Exception as exc:
        save_json(ARTIFACTS/"training_status.json",{"status":"error","message":str(exc)})
        raise
    finally:
        lock.unlink(missing_ok=True)
if __name__=="__main__":
    main()
