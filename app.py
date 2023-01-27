import json
import pickle

import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd
import requests


app = Flask(__name__)

# Load the model
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scaler = pickle.load(open('scaling.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json
    print("data",data,"\n")         #Values in 2D dictionary 

    nd = data.get("data")   # nd stores values of key "data" and type(nd)=dict
    print("nd",nd) 
    print("type", type(nd),"\n")

    li = list(nd.values())
    print(li,"\n", type(li),"\n")

    print(np.array(li).reshape(1, -1))
    new_data = scaler.transform(np.array(li).reshape(1,-1))
    print(new_data,"\n")
    output = regmodel.predict(new_data)
    print(output[0])

    return jsonify(output[0])

    # print(np.array(list(data.values())).reshape(1, -1))
    # new_data = scalar.transform(np.array(list(data.values())).reshape(1, -1))
    # output = regmodel.predict(li)
    # print(output[0])
    # return jsonify(output[0])


@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scaler.transform(np.array(data).reshape(1,-1))
    output = regmodel.predict(final_input)[0]
    return render_template("home.html", prediction_text="The House Price Predction in {}".format(output))


if __name__ == "__main__":
    app.run(debug=False)
