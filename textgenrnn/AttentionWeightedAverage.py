from keras.engine import InputSpec, Layer
from keras import backend as K
from keras import initializers


class AttentionWeightedAverage(Layer):
    def __init__(self, return_attention=False, **kwargs):
      self.init = initializers.get("uniform")
      self.supports_masking = True
      self.return_attention = return_attention
      super().__init__(**kwargs)

    def build(self, input_shape):
      self.input_spec = [InputSpec(ndim=3)]
      assert len(input_shape) == 3

      self.W = self.add_weight(shape=(input_shape[2], 1),
                               name='{}_W'.format(self.name),
                               initializer=self.init)
      self.trainable_weights = [self.W]
      super().build(input_shape)  
    
    def call(self, x, mask=None):
      pass
