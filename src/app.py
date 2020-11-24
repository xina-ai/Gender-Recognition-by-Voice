from xgboost import XGBClassifier
import flask
import config
import time
import pickle
import numpy as np
from flask import Flask, request
from utils import extract_features
from pydub import AudioSegment
import warnings
warnings.filterwarnings('ignore')
import base64
import io
import os

app = Flask(__name__)

MODEL = None


def audio_prediction(audio_data):
    data = base64.b64decode(audio_data)
    AudioSegment.from_file(io.BytesIO(data)).low_pass_filter(8000).set_frame_rate(48000).export(
        "tmp.mp3", format="mp3", bitrate='64k')
    feats = extract_features(
        'tmp.mp3', mel=True, mfcc=True, chroma=True, contrast=True)
    scaler = pickle.load(open(config.SCALAR_PATH, 'rb'))
    X = scaler.transform(feats.reshape(1, -1))
    pred = MODEL.predict_proba(X)
    try:
        os.remove('tmp.mp3')
    except:
        pass
    return pred[0][1]


@app.route("/predict", METHODS=['POST'])
def predict():
    audio_data = request.get_json()['audio']
    start_time = time.time()
    male_prediction = audio_prediction(audio_data)
    female_prediction = 1 - male_prediction
    response = {}
    response["response"] = {
        "male": str(male_prediction),
        "female": str(female_prediction),
        "time_taken": str(time.time() - start_time),
        'prediction': 'male' if male_prediction > female_prediction else 'female'
    }
    return flask.jsonify(response)


if __name__ == "__main__":
    MODEL = XGBClassifier()
    MODEL.load_model(config.MODEL_PATH)
    app.run()
