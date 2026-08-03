import io 
import json 
import urllib.request

import tensorflow


import numpy as np 

from PIL import Image 
from flask import Flask , request, jsonify
from flask_cors import CORS

from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)

from tensorflow.keras.preprocessing.image import img_to_array


app = Flask(__name__)
CORS(app)


# Loading the pre- train Data
model = MobileNetV2(weights='imagenet') 

top_k = 5
target_size = (224,224) 

@app.route("/predict", methods= ["POST"])

def predict():
    if "image" not in request.files:
        return jsonify({"error": " NO image file provided"}) , 400

    file = request.files["image"]

    try:
        img = Image.open(file.stream).convert("RGB")
    except Exception as e :
        return jsonify({"error": f"Invalid image : {str(e)}"}), 400

    # preprocessing the image
    img = img.resize(target_size)
    img_arry = img_to_array(img)
    img_dim = np.expand_dims(img_arry, axis=0)
    img_preprocessed  = preprocess_input(img_dim)

    # Interfernece
    predictions = model.predict(img_preprocessed)

    # Decoding top_k prediction i
    decoded = decode_predictions(predictions, top= top_k)[0]

    results = [{
        "label": label.replace("_"," "),
        "confidence" : round(float(confidence),4)
    }
        for(_, label, confidence) in decoded
    ]

    return jsonify({"predictions": results})

@app.route("/",methods= ["GET"])
def health_check():
    return jsonify({"status": "Backend is running "})

if __name__ =="__main__":
    app.run(debug=True , host="127.0.0.1", port= 5000)

    