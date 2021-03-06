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
    global prefix, temps, max_length, n_samples, weights_path, vocab_path, config_path
    max_length = int(request.form["max_length"])
    temps = [float(value) for value in request.form["temperatures"].split(',')]
    n_samples = int(request.form["samples"])
    weights_path = "textgenerator/weights/" + request.form["weightOption"] + "_weights.hdf5"
    vocab_path = "textgenerator/vocabs/" + request.form["weightOption"] + "_vocab.json"
    config_path = "textgenerator/configs/" + request.form["weightOption"] + "_config.json"
    prefix = request.form["prefix"]
    return redirect(url_for("index"))


@app.route("/read", methods=["GET", "POST"])
def read_file():
    try:
        assert max_length > 0
        assert len(temps) > 0
        assert n_samples > 0
        assert weights_path is not None
        assert vocab_path is not None
        assert config_path is not None
        train_gen = text_generator(
                weights_path=weights_path,
                vocab_path=vocab_path,
                config_path=config_path
            )
        gens = train_gen.generate_samples(n=n_samples, max_gen_length=max_length, temperatures=temps, return_as_list=True, prefix=prefix)
        print(gens)
        return render_template("text_generation.html", generated=gens, temps=temps)
    except Exception as e:
        return render_template("index.html", error=e)
    
    return render_template("index.html", error="Sucess")


if __name__ == "__main__":
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    set_session(sess)
    app.run(debug=True, port=5004, host='0.0.0.0')
    
