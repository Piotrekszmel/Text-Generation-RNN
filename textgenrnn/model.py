from keras.optimizers import RMSprop
from keras.layers import Input, Embedding, Dense, LSTM, Bidirectional
from keras.layers import concatenate, Reshape, SpatialDropout1D
from keras.models import Model
from keras import backend as K
from AttentionWeightedAverage import AttentionWeightedAverage


def textgenrnn_model(num_classes, cfg, context_size=None,
                     weights_path=None,
                     dropout=0.0,
                     optimizer=RMSprop(lr=4e-3, rho=0.99)):

    '''
    Builds the model architecture for textgenrnn and
    loads the specified weights for the model.
    '''

    input = Input(shape=(cfg["max_length"],), name="input")
    embedded = Embedding(num_classes, cfg["dim_embeddings"],
                        input_length=cfg["max_length"],
                        name="embedding")(input)

    if dropout > 0.0:
        embedded = SpatialDropout1D(dropout, name="dropout")(embedded)
    