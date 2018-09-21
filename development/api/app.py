# https://github.com/UnitedIncome/serverless-python-requirements
try:
  import unzip_requirements
except ImportError:
  pass

import base64
import os
import tempfile

import flask
from flask import Flask, request, jsonify
import numpy as np

import sys
#Insert path for development or include it in the Dockerfile
sys.path.insert(0, '')
from predict import predictor

'''
For Keras implementation see: https://github.com/gradescope/fsdl-text-recognizer-project/blob/master/lab6_sln/api/app.py
'''

app = Flask(__name__)

@app.route('/')
def index():
    return 'Hello, world'

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    data = _load_data()
    pred, conf = predictor.predict(data)
    return jsonify({'prediction': pred, 'confidence': conf})

def _load_data():
    if request.method == 'POST':
        data = request.get_json()
        if data is None:
            return 'No data received'
        return data
    elif request.method == 'GET':
        url = request.args.get('url')
        if url is None:
            return 'No URL'
        return url
    else:
        ValueError('Unsupported HTTP method')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)
