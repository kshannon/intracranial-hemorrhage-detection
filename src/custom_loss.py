import tensorflow as tf
from tensorflow import keras as K


def multilabel_loss(class_weights=None):
    def multilabel_loss_inner(y_true, logits):
        logits = tf.cast(logits, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        
        # compute single class cross entropies:
        contributions = tf.maximum(logits, 0) - tf.multiply(logits, y_true) + tf.log(1.+tf.exp(-tf.abs(logits)))

        # contributions have shape (n_samples, n_classes), we need to reduce with mean over samples to obtain single class xentropies:
        single_class_cross_entropies = tf.reduce_mean(contributions, axis=0)

        # if None, weight equally:
        if class_weights is None:
            loss = tf.reduce_mean(single_class_cross_entropies)
        else:
            weights = tf.constant(class_weights, dtype=tf.float32)
            loss = tf.reduce_sum(tf.multiply(weights, single_class_cross_entropies))
        return loss
    return multilabel_loss_inner

