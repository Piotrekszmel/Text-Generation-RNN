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


def text_generation_sample(preds, temperature, interactive=False):
    '''
    Samples predicted probabilities of the next character 
    '''
    preds = np.asarray(preds).astype('float64')

    if temperature is None or temperature == 0.0:
        return np.argmax(preds)

    preds = np.log(preds + K.epsilon()) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)

    
    index = np.argmax(probas)
    # prevent function from being able to choose 0 (placeholder)
    # choose 2nd best index from preds
    if index == 0:
       index = np.argsort(preds)[-2]
    return index


def text_generation_generate(model, vocab,
                        indices_char, temperature=0.5,
                        maxlen=40, meta_token='<s>',
                        word_level=False,
                        single_text=False,
                        max_gen_length=300,
                        prefix=None,
                        synthesize=False,
                        stop_tokens=[' ', '\n']):
    """
    Generates and returns a single text.

    Parameters:
    indices_char: dict mapping indcies to characters
    
    temperature: temperature or list of temperatures that will be used during generation
    
    maxlen: maximum number of characters that will be involved in predicting next character
    
    meta_token:

    single_text: 

    max_gen_length: maximum length of generated text

    prefix: Each generated text will start with a given text

    synthesize (Boolean): True if You use synthesize or synthesize method else False

    stop_tokens: Token that stop the generation. Informs the generator using the synthesize method that a word has been generated. Generator then changes the model for predicting next word
    """

    collapse_char = ' ' if word_level else ''
    end = False

    # If generating word level, must add spaces around each punctuation.
    if word_level and prefix:
        punct = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\\n\\t\'‘’“”’–—'
        prefix = re.sub('([{}])'.format(punct), r' \1 ', prefix)
        prefix_t = [x.lower() for x in prefix.split()]

    if not word_level and prefix:
        prefix_t = list(prefix)

    if single_text:
        text = prefix_t if prefix else ['']
        max_gen_length += maxlen
    else:
        text = [meta_token] + prefix_t if prefix else [meta_token]

    next_char = ''

    if not isinstance(temperature, list):
        temperature = [temperature]

    if len(model.inputs) > 1:
        model = Model(inputs=model.inputs[0], outputs=model.outputs[1])

    while not end and len(text) < max_gen_length:
        encoded_text = text_generation_encode_sequence(text[-maxlen:], vocab, maxlen)
        next_temperature = temperature[(len(text) - 1) % len(temperature)]

        next_index = text_generation_sample(model.predict(encoded_text, batch_size=1)[0], next_temperature)
        next_char = indices_char[next_index]
        text += [next_char]
        if next_char == meta_token or len(text) >= max_gen_length:
            end = True
        gen_break = (next_char in stop_tokens or word_level or
                     len(stop_tokens) == 0)
        if synthesize and gen_break:
            break
        

    # if single text, ignore sequences generated w/ padding
    # if not single text, remove the <s> meta_tokens
    if single_text:
        text = text[maxlen:]
    else:
        text = text[1:]
        if meta_token in text:
            text.remove(meta_token)

    text_joined = collapse_char.join(text)

    # If word level, remove spaces around punctuation for cleanliness.
    if word_level:
        punct = '\\n\\t'
        text_joined = re.sub(" ([{}]) ".format(punct), r'\1', text_joined)
        

    return text_joined, end


def text_generation_encode_sequence(text, vocab, maxlen):
    """
    Encodes a text into the corresponding encoding for prediction with
    the model.
    
    Parameters:
    text: text to encode
    vocab: char/word -> index vocabulary
    maxlen: max length of text to encode
    """
    encoded = np.array([vocab.get(x, 0) for x in text])
    return sequence.pad_sequences([encoded], maxlen=maxlen)


def text_generation_texts_from_file(file_path, header=True,
                               delim='\n', is_csv=False):
    """
    Retrieves texts from a newline-delimited file and returns as a list.
    """
    with open(file_path, 'r', encoding='utf8', errors='ignore') as f:
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


def text_generation_texts_from_file_context(file_path, header=True):
    """
    Retrieves texts+context from a two-column CSV.
    """

    with open(file_path, 'r', encoding='utf8', errors='ignore') as f:
        if header:
            f.readline()
        texts = []
        context_labels = []
        reader = csv.reader(f)
        for row in reader:
            texts.append(row[0])
            context_labels.append(row[1])

    return (texts, context_labels)


def text_generation_encode_cat(chars, vocab):
    """
    One-hot encodes values at given chars efficiently by preallocating
    a zeros matrix.
    """

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
            gen_text, end = text_generation_generate(textgen.model,
                                                textgen.vocab,
                                                textgen.indices_char,
                                                temperature,
                                                textgen.config['max_length'],
                                                textgen.META_TOKEN,
                                                textgen.config['word_level'],
                                                textgen.config.get(
                                                    'single_text', False),
                                                max_gen_length,
                                                prefix=gen_text,
                                                synthesize=True,
                                                stop_tokens=stop_tokens)
            textgen_i += 1
        if not return_as_list:
            print("{}\n".format(gen_text))
        gen_texts.append(gen_text)
    if return_as_list:
        return gen_texts


def synthesize_to_file(textgens, destination_path, **kwargs):
    """
    Save generated texts by synthesize to the given file
    """
    texts = synthesize(textgens, return_as_list=True, **kwargs)
    with open(destination_path, 'w') as f:
        for text in texts:
            f.write("{}\n".format(text))


class generate_after_epoch(Callback):
    def __init__(self, text_generation, gen_epochs, max_gen_length):
        self.text_generation = text_generation
        self.gen_epochs = gen_epochs
        self.max_gen_length = max_gen_length

    def on_epoch_end(self, epoch, logs={}):
        if self.gen_epochs > 0 and (epoch+1) % self.gen_epochs == 0:
            self.text_generation.generate_samples(
                max_gen_length=self.max_gen_length)


class save_model_weights(Callback):
    def __init__(self, text_generation, num_epochs, save_epochs):
        self.text_generation = text_generation
        self.weights_name = text_generation.config['name']
        self.num_epochs = num_epochs
        self.save_epochs = save_epochs

    def on_epoch_end(self, epoch, logs={}):
        if len(self.text_generation.model.inputs) > 1:
            self.text_generation.model = Model(inputs=self.model.input[0],
                                          outputs=self.model.output[1])
        if self.save_epochs > 0 and (epoch+1) % self.save_epochs == 0 and self.num_epochs != (epoch+1):
            print("Saving Model Weights — Epoch #{}".format(epoch+1))
            self.text_generation.model.save_weights(
                "weights/{}_weights_epoch_{}.hdf5".format(self.weights_name, epoch+1))
        else:
            self.text_generation.model.save_weights(
                "weights/{}_weights.hdf5".format(self.weights_name))