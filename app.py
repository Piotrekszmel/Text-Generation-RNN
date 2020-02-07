from flask import Flask, render_template, request, url_for, redirect
from tensorflow.python.keras.backend import set_session
import tensorflow as tf
import pandas as pd

import sys
sys.path.append("textgenerator/")
from textgenerator.text_generator import text_generator

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

app.route('/save_options', methods=["POST"])
def save_options():
    # TODO: change to post request
    global temps, max_length
    max_length = int(request.form["max_length"])
    temps = [float(value) for value in request.form["temperatures"].split(',')]
    return redirect(url_for("text_generation"))


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5004)