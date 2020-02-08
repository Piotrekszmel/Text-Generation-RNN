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


@app.route('/save_options', methods=["POST"])
def save_options():
    # TODO: change to post request
    global temps, max_length, weights_path, vocab_path, config_path
    max_length = int(request.form["max_length"])
    temps = [float(value) for value in request.form["temperatures"].split(',')]
    weights_path = "textgenerator/weights/" + request.form["weightOption"] + "_weights.hdf5"
    vocab_path = "textgenerator/vocabs/" + request.form["weightOption"] + "_vocab.json"
    config_path = "textgenerator/configs/" + request.form["weightOption"] + "_config.json"
    print(weights_path)
    print(vocab_path)
    print(config_path)
    return redirect(url_for("index"))


@app.route("/read", methods=["GET", "POST"])
def read_file():
    try:
        assert max_length > 0
        assert len(temps) > 0
        assert weights_path is not None
        assert vocab_path is not None
        assert config_path is not None
        train_gen = text_generator(
                weights_path=weights_path,
                vocab_path=vocab_path,
                config_path=config_path
            )
        gens = train_gen.generate_samples(max_gen_length=max_length, temperatures=temps, return_as_list=True)
        print(gens)
        return render_template("text_generation.html", generated=gens[0])
    except Exception as e:
        return render_template("index.html", error=e)
    
    return render_template("index.html", error="Sucess")

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5004)