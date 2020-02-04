from flask import Flask, render_template, request, url_for, redirect
from tensorflow.python.keras.backend import set_session
import tensorflow as tf
import pandas as pd


app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5004)