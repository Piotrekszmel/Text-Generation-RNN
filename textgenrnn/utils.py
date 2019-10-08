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
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)

    if not interactive:
      index = np.argmax(probas)

      if index == 0:
        index = np.argsort([preds])[-2]
      else:
        index = (-preds).argsort()[:top_n]
    
    return index


def textgenrnn_generate(model, vocab, indices_char, temperature=0.5, maxlen=40, meta_token="<s>",
                        word_level=False, single_text=False, max_gen_length=300, interactive=False,
                        top_n=3, prefix=None, synthesize=False, stop_tokens=[' ', '\n']):
    '''
    Generates and returns a single text.
    '''

    collapse_char = " " if word_level else ""
    end = False

    if word_level and prefix:
        punct = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\\n\\t\'‘’“”’–—'
        prefix = re.sub('([{}])'.format(punct), r' \1 ', prefix)
        prefix_t = [x.lower() for x in prefix.split()]

    if not word_level and prefix:
      prefix_t = list(prefix)

    if single_text: 
      text = prefix_t if prefix else [""]
      max_gen_length += maxlen
    
    if not isinstance(temperature, list):
      temperature = [temperature]

    if len(model.inputs) > 1:
      model = Model(inputs=model.inputs[0], outputs=model.outputs[1])

      
    
