import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class EmbeddingsExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, word_indices, max_lengths=None, add_tokens=None, unk_policy="random", **kwargs):
        """
        :param word_indices:
        :param max_lengths: list of integers indicating the max limit of words for each data list in X
        :param unk_policy: "random","zero"
        """
        self.word_indices = word_indices
        self.max_lengths = max_lengths
        self.add_tokens = add_tokens
        self.unk_policy = unk_policy
        self.hierarchical = kwargs.get("hierarchical", False)

    @staticmethod
    def sequences_to_fixed_length(X, length):
        Xs = np.zeros((X.size, length), dtype="int32")
        for i, x in enumerate(X):
            if x.size < length:
                Xs[i] = np.pad(x, (0, length - len(x) % length), "constant")
            elif x.size > length:
                Xs[i] = x[0:length]
        
        return Xs

    def get_fixed_size_topic(self, X, max_lengths):
        X = list(X)
        Xs = np.zeros((len(X), max_lengths), dtype="int32")
        
        for i, doc in enumerate(X):
            Xs[i, 0] = self.word_indices.get("<s>", 0)
            for j, token in enumerate(doc[:max_lengths]):
                if token in self.word_indices:
                    Xs[i, min(j + 1, max_lengths - 1)] = self.word_indices[token]

                else: 
                    if self.unk_policy == "random":
                        Xs[i, min(j + 1, max_lengths - 1)] = self.word_indices["<unk>"]
                    elif self.unk_policy == "zero":
                        Xs[i, min(j + 1, max_lengths - 1)] = 0

            if len(doc) + 1 < max_lengths:
                Xs[i, len(doc) + 1] = self.word_indices.get("</s>", 0)
        
        return Xs
    
    def index_text(self, sent, add_tokens=False):
        sent_words = []
        
        if add_tokens:
            sent_words.append(self.word_indices.get("<s>", 0))
        
        for token in sent:
            if token in self.word_indices:
                sent_words.append(self.word_indices[token])
            else:
                if self.unk_policy == "random":
                    sent_words.append(self.word_indices.get("<unk>", 0))
                elif self.unk_policy == "zero":
                    sent_words.append(0)
        
        if add_tokens:
            sent_words.append(self.word_indices.get("</s>", 0))
        
        return sent_words

    def words_to_indices(self, X, add_tokens=False):
        """
        :param X: list of texts
        :param add_tokens:
        """
        Xs = []
        if isinstance(X, list):
            for sent in X:
                Xs.append(np.asarray(self.index_text(sent, add_tokens=add_tokens)))
        else:
            Xs = np.asarray(self.index_text(X, add_tokens=add_tokens))
        return np.asarray(Xs)

    def index_text_list(self, texts, length, add_tokens):
        """
        Converts a list of texts (strings) to a list of lists of integers (word ids)
        :param texts: the list of texts
        :param length: the maximum length that a text can have. 0 means no limit
        :param add_tokens: whether to add special tokens in the beginning and at the end of each text
        :return: list of lists of integers (word ids)
        """
        indexed = self.words_to_indices(texts, add_tokens=add_tokens)

        if length > 0:
            indexed = self.sequences_to_fixed_length(indexed, length)
        
        return indexed
        
    def transform(self, X, y=None):
        if isinstance(X, str):
            pass
        else:
            X = list(X)
        print('Trans1', X)
        # if the input contains multiple texts eg. text and aspect
        if isinstance(X[0][0], list):

            if self.max_lengths is None:
                max_lengths = [0] * len(X)
            else:
                max_lengths = self.max_lengths
            if self.add_tokens is None:
                add_tokens = [False] * len(X)
            else:
                add_tokens = self.add_tokens

            if self.hierarchical:
                assert self.max_lengths is not None
                assert self.add_tokens is not None
                max_lengths = [self.max_lengths] * len(X)
                add_tokens = [self.add_tokens] * len(X)
                return ([self.index_text_list(texts, length, add_tokens)
                        for texts, length, add_tokens in zip(X, max_lengths, add_tokens)])
            else:

                return [self.index_text_list(texts, length, add_tokens)
                        for texts, length, add_tokens in zip(zip(*X), max_lengths, add_tokens)]

        else:
            if self.max_lengths is None:
                max_lengths = 0
            else:
                max_lengths = self.max_lengths

            if self.add_tokens is None:
                add_tokens = False
            else:
                add_tokens = self.add_tokens

            return self.index_text_list(X, max_lengths, add_tokens)

    
    def fit(self, X, y=None):
        return self 
