from flask import Flask, request, render_template
from flask_cors import cross_origin
import pickle
import sklearn
import pandas as pd
import numpy as np

app = Flask(__name__)
model = pickle.load(open("cred_rf_final.pkl", "rb"))


@app.route("/")
@cross_origin()
def home():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
@cross_origin()
def predict():
    if request.method == "POST":
        age = float(request.form['AGE'])
        pay1 = float(request.form['PAY_1'])
        amt1 = float(request.form['PAY_AMT1'])
        amt6 = float(request.form['PAY_AMT6'])
        amt2 = float(request.form['PAY_AMT2'])
        amt5 = float(request.form['PAY_AMT5'])
        amt3 = float(request.form['PAY_AMT3'])
        amt4 = float(request.form['PAY_AMT4'])
        pay2 = float(request.form['PAY_2'])
        pay3 = float(request.form['PAY_3'])

        values = np.array([[age, pay1, amt1, amt6, amt2, amt5,amt3, amt4, pay2, pay3]])
        prediction = model.predict(values)

    if prediction == 1:
        return render_template('index.html',
                               prediction_text="Defaulter".format(prediction))
    elif prediction == 0:
        return render_template('index.html',
                               prediction_text="Not Defaulter".format(prediction))

        #output = round(prediction[0], 2)

        #return render_template('index.html', prediction_text="Your Flight price is Rs. {}".format(output))

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
