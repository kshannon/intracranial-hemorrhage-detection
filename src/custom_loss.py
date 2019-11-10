# Loss functions taken/inspired from Akensert's kaggle kernel

import numpy as np
import tensorflow as tf
from tensorflow import keras as K



def weighted_log_loss(y_true, y_pred):
    """
    Can be used as the loss function in model.compile()
    ---------------------------------------------------
    """
    
    # class_weights = np.array([1., 1., 1., 1., 1., 2.])
    class_weights = np.array([1.])
    
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
    
    # class_weights = K.backend.variable([1., 1., 1., 1., 1. ,2.])
    class_weights = K.backend.variable([1.])
    
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
    # class_weights = [1., 1., 1., 1., 1., 2.]
    class_weights = [1.]
    
    epsilon = 1e-7
    
    preds = np.clip(preds, epsilon, 1-epsilon)
    loss = trues * np.log(preds) + (1 - trues) * np.log(1 - preds)
    loss_samples = np.average(loss, axis=1, weights=class_weights)

    return - loss_samples.mean()