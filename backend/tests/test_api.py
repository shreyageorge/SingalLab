import io
import json
import sys
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
sys.path.insert(0, str(Path(__file__).parents[1]))
import app as service
from predict import decode_image, image_array, inference_array, locate_sign
from class_names import CLASS_NAMES

@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(service, 'ARTIFACTS', tmp_path)
    monkeypatch.setattr(service.store, 'model', None)
    return service.app.test_client()

def png():
    data=io.BytesIO()
    Image.new('RGB',(64,48),(255,0,0)).save(data,format='PNG')
    data.seek(0)
    return data

def test_health_and_classes(client):
    assert client.get('/api/health').status_code == 200
    assert client.get('/api/health').json['model_available'] is False
    classes=client.get('/api/classes').json['classes']
    assert len(classes)==43
    assert [c['id'] for c in classes]==list(range(43))
    assert classes[14]['name']=='Stop'
    assert classes[42]['name']==CLASS_NAMES[42]

def test_mapping_matches_frontend():
    path=Path(__file__).parents[2]/'frontend/src/data/classes.json'
    assert [c['name'] for c in json.loads(path.read_text())]==CLASS_NAMES

def test_missing_upload(client):
    assert client.post('/api/predict').status_code==400

def test_corrupt_upload(client):
    assert client.post('/api/predict',data={'image':(io.BytesIO(b'not image'),'x.png')}).status_code==400

def test_oversized_upload(client):
    assert client.post('/api/predict',data={'image':(io.BytesIO(b'x'* (5*1024*1024)),'x.png')}).status_code==413

def test_missing_model(client):
    response=client.post('/api/predict',data={'image':(png(),'sign.png')})
    assert response.status_code==503
    assert response.json['code']=='MODEL_UNAVAILABLE'

def test_missing_artifacts(client):
    data=client.get('/api/metrics').json
    assert not data['history']['available']
    assert not data['evaluation']['available']
    assert not client.get('/api/confusion-matrix').json['available']
    assert not client.get('/api/model-info').json['available']
    assert client.get('/api/samples').json['samples']==[]

def test_disabled_training_and_routes(client):
    assert client.post('/api/train',json={'epochs':20}).status_code==405
    assert client.get('/api/train/status').json['mode']=='cli'
    assert client.get('/api/samples/99').status_code==404
    assert client.get('/api/not-real').status_code==404

def test_preprocessing():
    array=image_array(decode_image(png().read()))
    assert array.shape==(32,32,3)
    assert array.dtype==np.float32
    assert array[0,0].tolist()==[255,0,0] # normalized inside the model, not twice

def test_inference_crops_a_dominant_uncropped_sign():
    image=Image.new('RGB',(320,240),(24,30,40))
    sign=Image.new('RGB',(90,100),(220,20,20))
    image.paste(sign,(115,70))
    crop=locate_sign(image)
    assert crop.size[0]==crop.size[1]
    assert crop.size[0]<image.width
    assert inference_array(image).shape==(32,32,3)

def test_inference_leaves_ambiguous_and_plain_images_unchanged():
    plain=Image.new('RGB',(320,240),(120,120,120))
    assert locate_sign(plain) is plain
    multiple=Image.new('RGB',(320,240),(24,30,40))
    multiple.paste(Image.new('RGB',(50,55),(220,20,20)),(45,90))
    multiple.paste(Image.new('RGB',(50,55),(220,20,20)),(225,90))
    assert locate_sign(multiple) is multiple

def test_training_groups_frames_by_sign_track():
    from train import track_group
    assert track_group(35,'00035_00000_00001')==track_group(35,'00035_00000_00002')
    assert track_group(35,'00035_00000_00001')!=track_group(35,'00035_00001_00001')
    assert track_group(35,'single_image')=='35/single_image'

def test_training_split_keeps_each_track_together_and_each_class_present():
    from train import split_dataset
    labels=np.repeat(np.arange(43),4)
    groups=np.array([f'{label}/{track}' for label in range(43) for track in range(4)])
    train_indices,val_indices=split_dataset(labels,groups,.1)
    assert set(labels[train_indices])==set(range(43))
    assert set(labels[val_indices])==set(range(43))
    assert set(groups[train_indices]).isdisjoint(groups[val_indices])

def test_real_artifacts_only(client,monkeypatch,tmp_path):
    (tmp_path/'training_history.json').write_text(json.dumps({'loss':[1.2]}))
    assert client.get('/api/metrics').json['history']['data']['loss']==[1.2]

def test_actual_keras_inference(client,monkeypatch):
    tf=pytest.importorskip('tensorflow')
    from train import build_model
    # Untrained fixture solely for API integration testing. Never saved or used by the application.
    tf.keras.utils.set_random_seed(42)
    model=build_model(.001)
    monkeypatch.setattr(service.store,'model',model)
    expected=np.asarray(model(image_array(decode_image(png().read()))[None],training=False))[0]
    response=client.post('/api/predict',data={'image':(png(),'sign.png')})
    assert response.status_code==200
    d=response.json
    assert d['class_id']==int(expected.argmax())
    assert d['score']==pytest.approx(float(expected.max()))
    assert d['class_name']==CLASS_NAMES[d['class_id']]
    assert len(d['top_k'])==5
    assert d['inference_ms']>=0

def test_loader_contract(monkeypatch,tmp_path):
    tf=pytest.importorskip('tensorflow')
    import hashlib
    import model_loader
    from train import build_model
    monkeypatch.setattr(model_loader,'ARTIFACTS',tmp_path)
    model=build_model(.001)
    path=tmp_path/'traffic_sign_cnn.keras';model.save(path)
    metadata={'class_names':CLASS_NAMES,'preprocessing':'embedded_rescaling_1_255','model_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    (tmp_path/'model_metadata.json').write_text(json.dumps(metadata))
    store=model_loader.ModelStore()
    assert store.status=='ready'
    assert store.info()['parameters']==model.count_params()
    metadata['class_names']=list(reversed(CLASS_NAMES))
    (tmp_path/'model_metadata.json').write_text(json.dumps(metadata))
    assert model_loader.ModelStore().status=='error'

def test_ppm_preview(client):
    data=io.BytesIO()
    Image.new('RGB',(50,50),(255,255,255)).save(data,format='PPM'); data.seek(0)
    response=client.post('/api/preview',data={'image':(data,'sample.ppm')})
    assert response.status_code==200
    assert response.mimetype=='image/png'
    assert client.post('/api/preview').status_code==400
    assert client.post('/api/preview',data={'image':(io.BytesIO(b'bad'),'x.ppm')}).status_code==400
