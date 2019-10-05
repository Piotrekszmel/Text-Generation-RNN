from keras.optimizers import RMSprop
from keras.layers import Input, Embedding, Dense, LSTM, Bidirectional
from keras.layers import concatenate, Reshape, SpatialDropout1D
from keras.models import Model
from keras import backend as K
from AttentionWeightedAverage import AttentionWeightedAverage
from utils import new_rnn

def textgenrnn_model(num_classes, cfg, context_size=None,
                     weights_path=None,
                     dropout=0.0,
                     optimizer=RMSprop(lr=4e-3, rho=0.99)):
    '''
    Builds the model architecture for textgenrnn and
    loads the specified weights for the model.
    '''

    input = Input(shape=(cfg['max_length'],), name='input')
    embedded = Embedding(num_classes, cfg['dim_embeddings'],
                         input_length=cfg['max_length'],
                         name='embedding')(input)

    if dropout > 0.0:
        embedded = SpatialDropout1D(dropout, name='dropout')(embedded)

    rnn_layer_list = []
    for i in range(cfg['rnn_layers']):
        prev_layer = embedded if i is 0 else rnn_layer_list[-1]
        rnn_layer_list.append(new_rnn(cfg, i+1)(prev_layer))
    
    seq_concat = concatenate([embedded] + rnn_layer_list, name="rnn_concat")
    attention = AttentionWeightedAverage(name="attention")(seq_concat)
    output = Dense(num_classes, name="output", activation="softmax")(attention)

    if context_size is None:
      model = Model(inputs=[input], outputs=[output])
      if weights_path is not None:
        model.load_weights(weights_path, by_name=True)
      
      model.compile(loss="categorical_crossentropy", optimizer=optimizer)

    else:
      context_input = Input(shape=(context_size, ), name="context_input")
      context_reshape = Reshape((context_input, ), name="context_reshape")(context_input)
      merged = concatenate([attention, context_reshape], name="concat")
      main_output = Dense(num_classes, name="context_output", activation="softmax")(merged)

      model = Model(inputs=[input, context_input], outputs=[main_output, output])

      if weights_path is not None:
        model.load_weights(weights_path, by_name=True)
      
      model.compile(loss="categorical_crossentropy", optimizer=optimizer, loss_weights=[0.8, 0.2])
    
    return model