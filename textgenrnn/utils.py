from keras.layers import LSTM, Bidirectional
from keras.callbacks import LearningRateScheduler, Callback
from keras.models import Model, load_model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras import backend as K
from sklearn.preprocessing import LabelBinarizer
from random import shuffle
from tqdm import trange
import numpy as np
import json
import h5py
import csv
import re

def new_rnn(cfg, layer_num):
    use_cudnnlstm = K.backend() == 'tensorflow' and len(K.tensorflow_backend._get_available_gpus()) > 0
    if use_cudnnlstm:
        from keras.layers import CuDNNLSTM
        if cfg['rnn_bidirectional']:
            return Bidirectional(CuDNNLSTM(cfg['rnn_size'],
                                           return_sequences=True),
                                 name='rnn_{}'.format(layer_num))

        return CuDNNLSTM(cfg['rnn_size'],
                         return_sequences=True,
                         name='rnn_{}'.format(layer_num))
    else:
        if cfg['rnn_bidirectional']:
            return Bidirectional(LSTM(cfg['rnn_size'],
                                      return_sequences=True,
                                      recurrent_activation='sigmoid'),
                                 name='rnn_{}'.format(layer_num))

        return LSTM(cfg['rnn_size'],
                    return_sequences=True,
                    recurrent_activation='sigmoid',
                    name='rnn_{}'.format(layer_num))



def textgenrnn_sample(preds, temperature, interactive=False, top_n=3):
    preds = np.asarray(preds).astype("float64")
    if temperature is None or temperature == 0.0:
      return np.argmax(preds)

    preds = np.log(preds + K.epsilon()) / temperature