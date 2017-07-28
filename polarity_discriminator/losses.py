import tensorflow as tf
from keras.losses import categorical_crossentropy
from keras import backend as K


class SumPolarityLoss:
    """
    """
    def __init__(self):
        pass

    def compute_loss(self, y_true, y_pred):
        """ compute loss
        """
        maxs = tf.expand_dims(tf.reduce_max(y_true, axis=-1), 1)
        y_pred_masked = y_true / maxs * y_pred
        pred_sum = tf.reduce_sum(y_pred_masked, axis=-1)
        true_sum = tf.reduce_sum(y_true, axis=-1)
        loss = K.mean(K.square(pred_sum - true_sum), axis=-1)

        return loss
