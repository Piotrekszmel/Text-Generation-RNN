from flask import Flask, render_template, request, url_for, redirect
from tensorflow.python.keras.backend import set_session
import tensorflow as tf
import pandas as pd

import sys
sys.path.append("textgenerator/")

from textgenerator.text_generator import text_generator
from textgenerator.utils import text_generation_texts_from_file
