import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, json, render_template, jsonify


app = Flask(__name__)

model2 = pickle.load(open('model1.pkl', 'rb'))


@app.route("/")
def home():
    return render_template("crop.html")


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    State_Name = str(request.form['State_Name'])
    crop_year = int(request.form['crop_year'])
    Season = str(request.form['Season'])
    crop = str(request.form['crop'])
    area = float(request.form['area'])



    X = [[State_Name,crop_year,Season,crop,area]]

    my_prediction = model2.predict(X)
    r=my_prediction[0]

    return render_template("crop.html", **locals())

if __name__=="__main__":
    app.run(debug=True, port=7895)





