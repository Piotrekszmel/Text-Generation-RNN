import pickle
import os
import numpy as np
import sys
sys.path.append('..')
sys.path.append('.')
from keras.callbacks import ModelCheckpoint
from keras.layers import LSTM
from ..utilities.data_preparation import get_labels_to_categories_map, get_class_weights2, onehot_to_categories
from sklearn.metrics import f1_score, precision_score
from sklearn.metrics import recall_score
from ..data.data_loader import DataLoader
from ..models.nn_models import build_attention_RNN
from ..utilities.data_loader import get_embeddings, Loader, prepare_dataset
from kutilities.callbacks import MetricsCallback, PlottingCallback


np.random.seed(1337)


def Sentiment_Analysis(WV_CORPUS, WV_DIM, max_length, PERSIST, TRAIN=False, FINAL=True):
	"""
	##Final:
	- if FINAL == False,  then the dataset will be split in {train, val, test}
	- if FINAL == True,   then the dataset will be split in {train, val}.

	##PERSIST
	# set PERSIST = True, in order to be able to use the trained model later
	"""

	best_model = lambda: "model.hdf5"
	best_model_word_indices = lambda: "model_word_indices.pickle"

	############################################################################
	# LOAD DATA
	############################################################################

	embeddings, word_indices = get_embeddings(corpus=WV_CORPUS, dim=WV_DIM)

	if PERSIST:
		pickle.dump(word_indices, open(best_model_word_indices(), "wb"))
	
	loader = Loader(word_indices, text_lengths=max_length)

	if FINAL:
		print("\n > running in FINAL mode!\n")
		training, testing = loader.load_final()
	else:
		training, validation, testing = loader.load_train_val_test()
	
	print("Building NN Model...")

	############################################################################
	# NN MODEL
	############################################################################

	nn_model = build_attention_RNN(embeddings, classes=3, max_length=max_length,
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

	print(nn_model.summary())

	############################################################################
	# CALLBACKS
	############################################################################
	metrics = {
		"f1_pn": (lambda y_test, y_pred:
					f1_score(y_test, y_pred, average='macro',
							labels=[class_to_cat_mapping['positive'],
									class_to_cat_mapping['negative']])),
		"M_recall": (
			lambda y_test, y_pred: recall_score(y_test, y_pred, average='macro')),
		"M_precision": (
			lambda y_test, y_pred: precision_score(y_test, y_pred,
													average='macro'))
	}

	classes = ['positive', 'negative', 'neutral']
	class_to_cat_mapping = get_labels_to_categories_map(classes)
	cat_to_class_mapping = {v: k for k, v in
							get_labels_to_categories_map(classes).items()}

	_datasets = {}
	_datasets["1-train"] = training,
	_datasets["2-val"] = validation if not FINAL else testing
	if not FINAL:
		_datasets["3-test"] = testing

	metrics_callback = MetricsCallback(datasets=_datasets, metrics=metrics)
	plotting = PlottingCallback(grid_ranges=(0.5, 0.75), height=5,
								benchmarks={"SE17": 0.681})

	_callbacks = []
	_callbacks.append(metrics_callback)
	_callbacks.append(plotting)

	if PERSIST:
		checkpointer = ModelCheckpoint(filepath=best_model(),
										monitor='val.macro_recall', mode="max",
										verbose=1, save_best_only=True)
		_callbacks.append(checkpointer)
	
	############################################################################
	# APPLY CLASS WEIGHTS
	############################################################################
	if TRAIN:
		class_weights = get_class_weights2(onehot_to_categories(training[1]),
										smooth_factor=0)
		print("Class weights:",
			{cat_to_class_mapping[c]: w for c, w in class_weights.items()})

		history = nn_model.fit(training[0], training[1],
							validation_data=validation if not FINAL else testing,
							epochs=50, batch_size=50,
							class_weight=class_weights)

		pickle.dump(history.history,
              open("sentiment.pickle", "wb"))
  