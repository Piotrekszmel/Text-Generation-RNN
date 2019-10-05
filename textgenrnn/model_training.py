
from keras.callbacks import LearningRateScheduler, Callback
from keras.models import Model, load_model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.utils import Sequence
from keras import backend as K
#from .utils import textgenrnn_encode_cat
import numpy as np