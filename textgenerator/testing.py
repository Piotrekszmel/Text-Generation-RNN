from text_generator import text_generator
import glob
import os
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)


model_cfg = {
    'word_level': False,   # set to True if want to train a word-level model (requires more data and smaller max_length)
    'rnn_size': 128,   # number of LSTM cells of each layer (128/256 recommended)
    'rnn_layers': 4,   # number of LSTM layers (>=2 recommended)
    'rnn_bidirectional': True,   # consider text both forwards and backward, can give a training boost
    'max_length': 25,   # number of tokens to consider before predicting the next (20-40 for characters, 5-10 for words recommended)
    'max_words': 10000,   # maximum number of words to model; the rest will be ignored (word-level model only)
}
train_cfg = {
    'line_delimited': True,   # set to True if each text has its own line in the source file
    'num_epochs': 100,   # set higher to train the model for longer
    'gen_epochs': 20,   # generates sample text from model after given number of epochs
    'train_size': 0.8,   # proportion of input data to train on: setting < 1.0 limits model from learning perfectly
    'dropout': 0.0,   # ignore a random proportion of source tokens each epoch, allowing model to generalize better
    'validation': False,   # If train__size < 1.0, test on holdout dataset; will make overall training slower
    'is_csv': False   # set to True if file is a CSV exported from Excel/BigQuery/pandas
}

texts = []

for line in open(os.path.join("/home/pszmelcz/Desktop/projects/University-Project-NLP/text_generation/datasets", "english.txt"), "r", encoding="utf-8").readlines():
    texts.append(line)
    if len(texts) == 300000:
        break


with open('datasets/eng.txt', 'w') as f:
    for text in texts:
        f.write("{}".format(text))

model_name = 'english_128_100_4LSTM'
file_name = "datasets/eng.txt"
textgen = text_generator(name=model_name)
train_function = textgen.train_from_file if train_cfg['line_delimited'] else textgen.train_from_largetext_file
train_function(
    file_path=file_name,
    new_model=True,
    num_epochs=train_cfg['num_epochs'],
    gen_epochs=train_cfg['gen_epochs'],
    batch_size=1024,
    train_size=train_cfg['train_size'],
    dropout=train_cfg['dropout'],
    validation=train_cfg['validation'],
    is_csv=train_cfg['is_csv'],
    rnn_layers=model_cfg['rnn_layers'],
    rnn_size=model_cfg['rnn_size'],
    rnn_bidirectional=model_cfg['rnn_bidirectional'],
    max_length=model_cfg['max_length'],
    dim_embeddings=100,
    word_level=model_cfg['word_level'])

"""
textgen = text_generator(weights_path='weights/english_128_100_4LSTM_weights.hdf5',
                       vocab_path='vocabs/english_128_100_4LSTM_vocab.json',
                       config_path='configs/english_128_100_4LSTM_config.json')
                       
textgen.generate_samples(max_gen_length=150, temperatures=[0.2, 0.5, 1], prefix="I am")
"""
