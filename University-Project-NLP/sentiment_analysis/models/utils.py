from .nn_models import build_attention_RNN
import sys
sys.path.append("..")
from ..utilities.data_loader import get_embeddings
from keras.layers import LSTM


def create_model(corpus, dim, weights_path=None):
    embeddings, word_indices = get_embeddings(corpus=corpus, dim=dim)
    nn_model = build_attention_RNN(embeddings, classes=3, max_length=50,
                                unit=LSTM, layers=2, cells=150,
                                bidirectional=True,
                                attention="simple",
                                noise=0.3,
                                final_layer=False,
                                dropout_final=0.5,
                                dropout_attention=0.5,
                                dropout_words=0.3,
                                dropout_rnn=0.3,
                                dropout_rnn_U=0.3,
                                clipnorm=1, lr=0.001, loss_l2=0.0001)
    
    if weights_path:
        nn_model.load_weights(weights_path)
    else:
        nn_model.build()
    return nn_model, word_indices
    
