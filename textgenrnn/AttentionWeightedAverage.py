from keras.engine import InputSpec, Layer
from keras import backend as K
from keras import initializers


class AttentionWeightedAverage(Layer):
    def __init__(self, return_attention=False, **kwargs):
        self.init = initializers.get("uniform")
        self.supports_masking = True
        self.return_attention = return_attention
        super(AttentionWeightedAverage, self).__init__(**kwargs)