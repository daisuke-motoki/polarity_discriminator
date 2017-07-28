import logging
import numpy as np
from keras.models import Model
from keras.layers import Flatten, LSTM, Dense, Input, Reshape
from keras.layers import BatchNormalization, Activation, Dropout, Embedding
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.merge import concatenate
from keras import regularizers
from polarity_discriminator.layers import L2Normalization

logger = logging.getLogger(__name__)


def network_model1(network_architecture):
    """ Model1
        Embedding -> LSMT -> FC
    Args:
        network_architecture: dict: network parameters
            max_sequence_len: max length of sentence
            n_word: number of words in dictionary
            word_dim: dimention of word embedding
            n_lstm_unit1: LSTMのユニット数
            rate_lstm_drop: LSTMのdropout率
    """
    # parameters
    max_sequence_len = network_architecture["max_sequence_len"]
    n_word = network_architecture["n_word"]
    word_dim = network_architecture["word_dim"]
    n_lstm_unit1 = network_architecture["n_lstm_unit1"]
    rate_lstm_drop = network_architecture["rate_lstm_drop"]

    # input
    inputs = Input(shape=(max_sequence_len,),
                   name="input_1")

    # embedding layer
    embed = Embedding(n_word,
                      word_dim,
                      input_length=max_sequence_len,
                      name="embed_{}_{}".format(n_word, word_dim)
                      )(inputs)
    # LSTM
    net = Bidirectional(
        LSTM(n_lstm_unit1,
             dropout=rate_lstm_drop,
             recurrent_dropout=rate_lstm_drop,
             return_sequences=True),
        name="bi-lstm_1"
    )(embed)

    net = Bidirectional(
        LSTM(n_lstm_unit1,
             dropout=rate_lstm_drop,
             recurrent_dropout=rate_lstm_drop,
             return_sequences=True),
        name="bi-lstm_2"
    )(net)

    prediction = TimeDistributed(
        Dense(1,
              # activity_regularizer=regularizers.l1(0.1),
              activation=None),
        name="time-dist_1"
    )(net)

    prediction = Flatten()(prediction)
    # prediction
    # prediction = L2Normalization(10, 2,
    #                              name="time-dist_1-norm")(net)
    model = Model(inputs, prediction)
    model.summary()

    return model
