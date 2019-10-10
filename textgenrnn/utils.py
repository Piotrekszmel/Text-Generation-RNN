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
    
    else:
      text = [meta_token] + prefix_t if prefix else [meta_token]
    
    next_char = ""
    
    if not isinstance(temperature, list):
      temperature = [temperature]

    if len(model.inputs) > 1:
      model = Model(inputs=model.inputs[0], outputs=model.outputs[1])

    while not end and len(text) < max_gen_length:
      encoded_text = textgenrnn_encode_sequence(text[-maxlen:], vocab, maxlen)

      next_temperature = temperature[(len(text) - 1) % len(temperature)]
      


def textgenrnn_encode_sequence(text, vocab, maxlen):
    '''
    Encodes a text into the corresponding encoding for prediction with
    the model.
    '''

    encoded = np.array([vocab.get(x, 0) for x in text])
    return sequence.pad_sequences([encoded], maxlen=maxlen)


def textgenrnn_texts_from_file(file_path, header=True, delim="\n", is_csv=False):
    """
    Retrieves texts from a newline-delimited file and returns as a list.
    """

    with open(file_path, 'r', encoding="utf8", errors="ignore") as f:
      if header:
        f.readline()
      
      if is_csv:
        texts = []
        reader = csv.reader(f)
        for row in reader:
          texts.append(row[0])

      else:
        texts = [line.rstrip(delim) for line in f]
    
    return texts


def textgenrnn_encode_cat(chars, vocab):
  a = np.float32(np.zeros((len(chars), len(vocab) + 1)))
  rows, cols = zip(*[(i, vocab.get(char, 0))
                       for i, char in enumerate(chars)])

  a[rows, cols] = 1
  return a 


def synthesize(textgens, n=1, return_as_list=False, prefix='',
               temperature=[0.5, 0.2, 0.2], max_gen_length=300,
               progress=True, stop_tokens=[' ', '\n']):
    """
    Synthesizes texts using an ensemble of input models.
    """

    gen_texts = []
    iterable = trange(n) if progress and n > 1 else range(n)
    for _ in iterable:
      shuffle(textgens)
      gen_text = prefix
      end = False
      textgen_i = 0
      while not end:
        textgen = textgens[textgen_i % len(textgens)]
        gen_text, end = textgenrnn_generate(textgen.model,
                                            textgen.vocab,
                                            textgen.indices_char, 
                                            temperature,
                                            textgen.config["max_length"],
                                            textgen.config.get("single_text", False),
                                            max_gen_length,
                                            prefix=gen_text,
                                            synthesize=True,
                                            stop_tokens=stop_tokens)
        textgen_i += 1
      
      if not return_as_list:
        print("{}.\n".format(gen_text))
      gen_texts.append(gen_text)
    
    if return_as_list:
      return gen_texts



  


    
