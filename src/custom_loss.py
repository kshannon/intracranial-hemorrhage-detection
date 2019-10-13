# Loss functions taken/inspired from Akensert's kaggle kernel

import numpy as np
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


def weighted_log_loss(y_true, y_pred):
    """
    Can be used as the loss function in model.compile()
    ---------------------------------------------------
    """
    
    class_weights = np.array([1., 1., 1., 1., 1., 2.])
    
    eps = K.backend.epsilon()
    
    y_pred = K.backend.clip(y_pred, eps, 1.0-eps)

    out = -(         y_true  * K.backend.log(      y_pred) * class_weights
            + (1.0 - y_true) * K.backend.log(1.0 - y_pred) * class_weights)
    
    return K.backend.mean(out, axis=-1)


def _normalized_weighted_average(arr, weights=None):
    """
    A simple Keras implementation that mimics that of 
    numpy.average(), specifically for the this competition
    """
    
    if weights is not None:
        scl = K.backend.sum(weights)
        weights = K.backend.expand_dims(weights, axis=1)
        return K.backend.sum(K.backend.dot(arr, weights), axis=1) / scl
    return K.backend.mean(arr, axis=1)


def weighted_loss(y_true, y_pred):
    """
    Will be used as the metric in model.compile()
    ---------------------------------------------
    
    Similar to the custom loss function 'weighted_log_loss()' above
    but with normalized weights, which should be very similar 
    to the official competition metric:
        https://www.kaggle.com/kambarakun/lb-probe-weights-n-of-positives-scoring
    and hence:
        sklearn.metrics.log_loss with sample weights
    """
    
    class_weights = K.backend.variable([1., 1., 1., 1., 1. ,2.])
    
    eps = K.backend.epsilon()
    
    y_pred = K.backend.clip(y_pred, eps, 1.0-eps)

    loss = -(        y_true  * K.backend.log(      y_pred)
            + (1.0 - y_true) * K.backend.log(1.0 - y_pred))
    
    loss_samples = _normalized_weighted_average(loss, class_weights)
    
    return K.backend.mean(loss_samples)


def weighted_log_loss_metric(trues, preds):
    """
    Will be used to calculate the log loss 
    of the validation set in PredictionCheckpoint()
    ------------------------------------------
    """
    class_weights = [1., 1., 1., 1., 1., 2.]
    
    epsilon = 1e-7
    
    preds = np.clip(preds, epsilon, 1-epsilon)
    loss = trues * np.log(preds) + (1 - trues) * np.log(1 - preds)
    loss_samples = np.average(loss, axis=1, weights=class_weights)

    return - loss_samples.mean()