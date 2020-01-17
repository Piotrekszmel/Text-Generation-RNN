from flask import Flask, render_template, request, url_for, redirect
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
import pandas as pd
import sys
sys.path.append("text_generation/")

from sentiment_analysis.evaluate import predict_sentiment_single_tweet 
from sentiment_analysis.models.utils import create_model
from sentiment_analysis.utilities.data_loader import Loader

from text_generation.text_generator import text_generator
from text_generation.utils import (text_generation_texts_from_file,
        text_generation_texts_from_file_context)


TEXT_PATH = "text_generation/data/texts/"
app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
  return render_template("index.html")

@app.route("/Sentiment", methods=["GET", "POST"])
def sentiment():
    if request.method == "POST":
        message = request.form["text"]
        with graph.as_default():
            set_session(sess)
            prediction = predict_sentiment_single_tweet(message, sentiment_model, loader.pipeline)
            return render_template('sentiment.html', prediction=prediction[0])
    return render_template("sentiment.html")
      

@app.route("/Generation", methods=['GET', 'POST'])
def text_generation():
    return render_template("text_generation.html")


@app.route('/save_options', methods=["POST"])
def save_options():
    # TODO: change to post request
    global temps, max_length
    max_length = int(request.form["max_length"])
    temps = [float(value) for value in request.form["temperatures"].split(',')]
    return redirect(url_for("text_generation"))

@app.route("/read", methods=["GET", "POST"])
def read_file():
    #f = request.form["data_file"]
    try:
        assert max_length > 0
        assert len(temps) > 0
        f = "test_text.txt"
        file_path = TEXT_PATH + f
        train_gen = text_generator(
                weights_path="text_generation/weights/english_128_300LSTM_weights.hdf5",
                vocab_path="text_generation/vocabs/english_128_300LSTM_vocab.json",
                config_path="text_generation/configs/english_128_300LSTM_config.json"
            )
        gens = train_gen.generate_samples(max_gen_length=max_length, temperatures=temps, return_as_list=True)
        print(gens)
        return str(gens)
    except Exception as e:
        return render_template("text_generation.html", error=e)
    """
    text = []
    try:
        for line in f:
            line = line.decode("latin-1").replace("\r", "").replace("\n", "")
            text.append(line)
    except Exception as e:
        print(e)
    print(text)
    """
    return render_template("text_generation.html", error="Sucess")


if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    set_session(sess)
    global graph, sentiment_model
    sentiment_model, word_indices = create_model("datastories.twitter", 300, "sentiment_analysis/weights/bi_model_weights_1.h5")
    graph = tf.get_default_graph()
    
    loader = Loader(word_indices, text_lengths=50)
    
    app.run(debug=False, host="0.0.0.0", port=5004)
